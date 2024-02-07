INDEX=0
for TOWN in 1 2 3 4 5 6 7
do
    for DATA in 0 1 2
    do
        echo "[${INDEX}] Collecting data: Town0 ${TOWN} Round ${DATA}"
        python main.py --town ${TOWN} --index ${INDEX} --frame 300
        INDEX=$((INDEX+1))
    done
done
