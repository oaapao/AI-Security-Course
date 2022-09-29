export PYTHONPATH=$(pwd):$(pwd)/src:$PYTHONPATH
python src/evaluate.py --task_id 1 --gt gt.txt