export PYTHONPATH=$(pwd):$(pwd)/src:$PYTHONPATH
python src/test.py --task_id 1 --epoch 20 --device cuda:3