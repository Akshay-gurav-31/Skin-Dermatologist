<?php
// Indexed arrays
$items = array("Pen", "Book", "Pencil");
$prices = array(10, 50, 5);

$total = 0;

// Using a simple for loop
for ($i = 0; $i < count($items); $i++) {
    echo $items[$i] . " : Rs." . $prices[$i] . "<br>";
    $total += $prices[$i];
}

echo "<br>Total Cost = Rs.$total";

// Control structure
if ($total > 50) {
    echo "<br>You purchased many items!";
} else {
    echo "<br>Small purchase.";
}
?>
