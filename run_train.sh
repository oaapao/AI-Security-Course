export PYTHONPATH=$(pwd):$(pwd)/src:$PYTHONPATH
python src/train.py --task_id 1 --epoch 20 --lr 0.0001 --device cuda:3