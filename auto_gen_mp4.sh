INDEX=0
while IFS= read -r d;
do
    while IFS= read -r dd;
    do
        # echo "Generating ${INDEX}th Video: ${dd}"
        printf -v INDEX_FILE "%04d" $INDEX
        echo ffmpeg -y -framerate 10 -i "${dd}/%4d.png" "./data/gt/carla/carla_2d_box_train/${INDEX_FILE}.mp4" >> generate_mp4.sh
        INDEX=$((INDEX+1))
    done <<<$(find ${d}/image/* -prune -type d)
done <<<$(find ./data/gt/carla/carla_2d_box_train/* -prune -type d)

chmod u+x generate_mp4.sh
./generate_mp4.sh

