echo "downloading models (this may take a while)..."
$ProgressPreference = 'SilentlyContinue'

wget https://github.com/cansik/yolo-hand-detection/releases/download/pretrained/cross-hands.cfg -O cross-hands.cfg
wget https://github.com/cansik/yolo-hand-detection/releases/download/pretrained/cross-hands.weights -O cross-hands.weights

echo "done!"