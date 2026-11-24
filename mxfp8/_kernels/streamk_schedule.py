"""Host-only immutable hybrid Stream-K work partition; built before capture."""
def schedule(n, k, m, sms, raster_group=13, tail_waves=1, *, tile_rows=128, tile_cols=128, block_k=128):
    """Keep full raster waves; partition the remaining K-block work contiguously.

    tail_waves=0 splits only the incomplete wave, 1 also includes the preceding
    full wave. An exactly full grid always stays data parallel.
    """
    if min(n,k,m,sms,raster_group)<=0 or k%tile_cols or m%block_k or tail_waves<0:
        raise ValueError('Expected positive config-12 dimensions and nonnegative tail_waves')
    nr, nc, steps = (n+tile_rows-1)//tile_rows, k//tile_cols, m//block_k
    total = nr*nc
    grid = min(sms,total*steps)
    group = min(raster_group,nr)
    dp = total if total%grid==0 else max(0,total//grid-tail_waves)*grid
    tail_work = (total-dp)*steps
    works, offsets, splits = [], [], {}
    coverage = [[] for _ in range(total)]
    slots, loads = 0, []

    def coord(task):
        first = task//(group*nc)*group
        rows = min(group,nr-first)
        local = task%(group*nc)
        return (first+local%rows)*tile_rows, (local//rows)*tile_cols

    def add(task, begin, end):
        nonlocal slots
        row,col = coord(task)
        slot = -1
        if begin or end!=steps:
            slot=slots;slots+=1
            splits.setdefault(task,[]).append((begin,slot))
        works.append([row,col,begin,end,slot])
        coverage[task].append((begin,end,slot))

    for cta in range(grid):
        offsets.append(len(works))
        load=0
        for task in range(cta,dp,grid):
            add(task,0,steps);load+=steps
        u,stop=cta*tail_work//grid,(cta+1)*tail_work//grid
        while u<stop:
            task=dp+u//steps;begin=u%steps;end=min(steps,begin+stop-u)
            add(task,begin,end);load+=end-begin;u+=end-begin
        loads.append(load)
    offsets.append(len(works))
    fixups, contributors = [], []
    for task,pieces in sorted(splits.items()):
        assert len(pieces)>=2
        row,col=coord(task)
        fixups.append([row,col,len(contributors),len(pieces)])
        contributors.extend(slot for _,slot in sorted(pieces))
    # Exact coverage and one writer per full tile/partial slot are plan invariants.
    for pieces in coverage:
        cursor=0
        for begin,end,_ in sorted(pieces):
            assert begin==cursor and begin<end<=steps
            cursor=end
        assert cursor==steps
    assert slots<=2*grid and len(contributors)==slots
    return dict(work=works, offsets=offsets, fixups=fixups, contributors=contributors,
                info=dict(grid=grid,total_tiles=total,steps=steps,dp_tiles=dp,
                          sk_tiles=total-dp,partial_slots=slots,fixup_tiles=len(fixups),
                          work_records=len(works),cta_steps=loads,raster_group=group,
                          tail_waves=tail_waves,workspace_bytes=max(1,slots)*tile_rows*tile_cols*4))

