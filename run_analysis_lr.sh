export PYTHONPATH=$(pwd):$(pwd)/src:$PYTHONPATH

# change learning rate
lr=(0.0001 0.0002 0.0005 0.001 0.002 0.005 0.01 0.02 0.05 0.1)
for ((i = 1; i <= 10; i++)); do
  mkdir src/tasks/task-$i
  python src/train.py --task_id $i --epoch 30 --lr ${lr[$i - 1]} --device cuda:3 >>src/tasks/task-$i/output.log
  python src/test.py --task_id $i --epoch 30 --device cuda:3 >>src/tasks/task-$i/output.log
  python src/evaluate.py --task_id $i --gt gt.txt >>src/tasks/task-$i/output.log
done
