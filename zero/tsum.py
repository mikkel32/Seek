"""one line per finished test of a log written by gt.sh / stest.sh:  python3 tsum.py <log> ..."""
import json, re, sys
for path in sys.argv[1:]:
    for l in open(path):
        m = re.match(r'(\[[0-9:]+\]) (.*?): (\{.*)', l.strip())
        if not m:
            continue
        try:
            d = json.loads(m.group(3))
        except Exception:
            print(l[:160].strip()); continue
        tl = (d.get('our_time_losses') or 0) + (d.get('their_time_losses') or 0) + (d.get('illegal') or 0)
        print(f'{m.group(1)} {m.group(2)[:78]:78s} {d["n"]:5d} games  {d["w"]}-{d["d"]}-{d["l"]}  Elo {d["elo"]:+.1f} ({d["lo"]:+.1f} to {d["hi"]:+.1f})' + (f'  time/illegal losses {tl}' if tl else '') + (f'  [{d["verdict"]}]' if d.get('verdict') else ''))
