#! /bin/bash
for lang in "hin" "ben" "mni" "multi"
do
    export LANG=$lang
    echo "LANGUAGE IS : $LANG"
    python comma_icon/train.py
    echo "$LANG TRAINING DONE..TESTING..."
    python comma_icon/generate_test.py
    echo "$LANG TESTING DONE..."
done

echo "COMPLETED"