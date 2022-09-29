export PYTHONPATH=$(pwd):$(pwd)/src:$PYTHONPATH
python src/test.py --task_id 1 --epoch 5 --device cuda:3