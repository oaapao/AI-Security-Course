export PYTHONPATH=$(pwd):$(pwd)/src:$PYTHONPATH
python src/evaluate.py --gt gt.txt --pre prediction.txt