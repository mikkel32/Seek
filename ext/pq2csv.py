"""One parquet file of the Hugging Face copy of the Lichess evaluation database -> rows fen,first move,depth,cp,mate on standard output."""
import sys
import pyarrow as pa, pyarrow.compute as pc, pyarrow.csv as csv, pyarrow.parquet as pq
out = sys.stdout.buffer; opt = csv.WriteOptions(include_header=False, quoting_style='none')
for b in pq.ParquetFile(sys.argv[1]).iter_batches(batch_size=1_000_000, columns=['fen', 'line', 'depth', 'cp', 'mate']):
    csv.write_csv(pa.table({'fen': b.column('fen'), 'mv': pc.utf8_slice_codeunits(b.column('line'), 0, 5), 'depth': b.column('depth'), 'cp': b.column('cp'), 'mate': b.column('mate')}), out, opt)
