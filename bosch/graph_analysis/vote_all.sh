FILENAME=$1
THRESHOLD=$2

for X in $(seq 0 23)
do
  python ../graph_analysis/vote.py $FILENAME -t $THRESHOLD -v $X
done
