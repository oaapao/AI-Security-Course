export PYTHONPATH=$(pwd):$(pwd)/src:$PYTHONPATH
python src/train.py --epoch 20 --lr 0.0001 --path weights --device cuda