path="$1_$2"
python createsegments.py $1 $2
th neural.lua "InputContentSegments/${path}_0.jpg" "InputStyleSegments/${path}_0.jpg" "TemporaryResults/${path}_0.jpg"
th neural.lua "InputContentSegments/${path}_1.jpg" "InputStyleSegments/${path}_1.jpg" "TemporaryResults/${path}_1.jpg"
th neural.lua "InputContentSegments/${path}_2.jpg" "InputStyleSegments/${path}_2.jpg" "TemporaryResults/${path}_2.jpg"
th neural.lua "InputContentSegments/${path}_3.jpg" "InputStyleSegments/${path}_3.jpg" "TemporaryResults/${path}_3.jpg"
th neural.lua "InputContentSegments/${path}_4.jpg" "InputStyleSegments/${path}_4.jpg" "TemporaryResults/${path}_4.jpg"
th neural.lua "InputContentSegments/${path}_5.jpg" "InputStyleSegments/${path}_5.jpg" "TemporaryResults/${path}_5.jpg"
python combinesegments.py $1 $2
