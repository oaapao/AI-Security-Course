export PYTHONPATH=$(pwd):$(pwd)/src:$PYTHONPATH

# change learning rate
lr=(0.0001 0.0002 0.0003 0.0004 0.0005 0.0006 0.0007 0.0008 0.0009 0.001)
for ((i = 1; i <= 10; i++)); do
  mkdir src/tasks/task-$i
  python src/train.py --task_id $i --epoch 30 --lr ${lr[$i - 1]} --device cuda:3 >>src/tasks/task-$i/output.log
  python src/test.py --task_id $i --epoch 30 --device cuda:3 >>src/tasks/task-$i/output.log
  python src/evaluate.py --task_id $i --gt gt.txt >>src/tasks/task-$i/output.log
done

for ((i = 11; i <= 14; i++)); do
  mkdir src/tasks/task-$i
done
# change model architecture(whether to use BN and dropout)
python src/train.py --task_id 11 --epoch 30 --lr 0.0003 --device cuda:2 >>src/tasks/task-11/output.log
python src/train.py --task_id 12 --epoch 30 --lr 0.0003 --device cuda:2 --no_dropout >>src/tasks/task-12/output.log
python src/train.py --task_id 13 --epoch 30 --lr 0.0003 --device cuda:2 --no_bn >>src/tasks/task-13/output.log
python src/train.py --task_id 14 --epoch 30 --lr 0.0003 --device cuda:2 --no_dropout --no_bn >>src/tasks/task-14/output.log

for ((i = 11; i <= 14; i++)); do
  python src/test.py --task_id $i --epoch 30 --device cuda:3 >>src/tasks/task-$i/output.log
  python src/evaluate.py --task_id $i --gt gt.txt >>src/tasks/task-$i/output.log
done
