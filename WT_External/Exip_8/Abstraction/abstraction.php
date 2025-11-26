<?php
abstract class Shape {
    abstract public function area(); // No body
}

class Square extends Shape {
    public function area() {
        echo "Area = side * side<br>";
    }
}

$s = new Square();
$s->area();
?>
