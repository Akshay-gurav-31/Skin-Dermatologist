<?php
class Student {
    public $name;
    public $age;

    public function display() {
        echo "Name: $this->name, Age: $this->age";
    }
}

// Creating objects
$s1 = new Student();
$s1->name = "Akash";
$s1->age = 21;

$s1->display();
?>
