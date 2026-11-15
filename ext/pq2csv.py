"""One parquet file -> rows fen,first move,depth,cp,mate on standard output (the input of ext/lconv in csv mode).
  python3 ext/pq2csv.py file.parquet            the Lichess evaluation database (columns fen, line, depth, cp, mate; several rows a position)
  python3 ext/pq2csv.py file.parquet fishnet    Lichess game positions with server analysis (columns fen, cp, mate, move = the move the player made)"""
import sys
import pyarrow as pa, pyarrow.compute as pc, pyarrow.csv as csv, pyarrow.parquet as pq
out = sys.stdout.buffer; opt = csv.WriteOptions(include_header=False, quoting_style='none'); fish = len(sys.argv) > 2 and sys.argv[2] == 'fishnet'
for b in pq.ParquetFile(sys.argv[1]).iter_batches(batch_size=1_000_000, columns=['fen', 'cp', 'mate', 'move'] if fish else ['fen', 'line', 'depth', 'cp', 'mate']):
    if fish:
        t = pa.table({'fen': b.column('fen'), 'mv': pc.fill_null(b.column('move'), ''), 'depth': pa.array([1] * b.num_rows, pa.int8()), 'cp': b.column('cp'), 'mate': b.column('mate')})
    else:
        t = pa.table({'fen': b.column('fen'), 'mv': pc.utf8_slice_codeunits(b.column('line'), 0, 5), 'depth': b.column('depth'), 'cp': b.column('cp'), 'mate': b.column('mate')})
    csv.write_csv(t, out, opt)
