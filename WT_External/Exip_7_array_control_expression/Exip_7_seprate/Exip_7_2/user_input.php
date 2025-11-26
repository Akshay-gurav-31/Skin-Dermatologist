<!DOCTYPE html>
<html>
<body>

<form method="post">
    Enter Fruit 1: <input type="text" name="f1"><br><br>
    Enter Fruit 2: <input type="text" name="f2"><br><br>
    Enter Fruit 3: <input type="text" name="f3"><br><br>

    <input type="submit" value="Show Fruits">
</form>

<?php
if ($_SERVER["REQUEST_METHOD"] == "POST") {

    // Taking user input in an array
    $fruits = array($_POST['f1'], $_POST['f2'], $_POST['f3']);

    // Display elements
    echo "First fruit: " . $fruits[0] . "<br>";
    echo "Second fruit: " . $fruits[1] . "<br>";
    echo "Third fruit: " . $fruits[2];
}
?>

</body>
</html>
