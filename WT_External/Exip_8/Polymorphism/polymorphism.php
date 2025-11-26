<?php
class ParentClass {
    public function greet() {
        echo "Hello from Parent<br>";
    }
}

class ChildClass extends ParentClass {
    public function greet() {
        echo "Hello from Child<br>";
    }
}

$c = new ChildClass();
$c->greet();
?>
