<!DOCTYPE html>
<html>
<body>

<form method="post">
    Enter Item 1 Name: <input type="text" name="item1"><br><br>
    Enter Item 1 Price: <input type="number" name="price1"><br><br>

    Enter Item 2 Name: <input type="text" name="item2"><br><br>
    Enter Item 2 Price: <input type="number" name="price2"><br><br>

    Enter Item 3 Name: <input type="text" name="item3"><br><br>
    Enter Item 3 Price: <input type="number" name="price3"><br><br>

    <input type="submit" value="Calculate Total">
</form>

<?php
if ($_SERVER["REQUEST_METHOD"] == "POST") {

    // Take inputs from user
    $items = array($_POST['item1'], $_POST['item2'], $_POST['item3']);
    $prices = array($_POST['price1'], $_POST['price2'], $_POST['price3']);

    $total = 0;

    // For loop
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
}
?>

</body>
</html>
