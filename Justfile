run:
  uv run main.py -o out -u "00ef538e88634ddd9810d034b748c24d" -q 10

pack:
  mkdir -p release
  mkdir -p release/rpt
  mkdir -p release/src
  typst ./report.typ -o release/rpt/report.pdf
  cp -r out release/out

clean:
  rm -r out
  rm -r release
