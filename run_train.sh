export PYTHONPATH=$(pwd):$(pwd)/src:$PYTHONPATH
python src/train.py --task_id 20 --epoch 1 --lr 0.0001 --device cuda:3
