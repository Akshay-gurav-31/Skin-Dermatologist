<!DOCTYPE html>
<html>
<body>

<form method="post">
    Enter your marks: 
    <input type="number" name="marks">
    <br><br>

    <input type="submit" value="Check Result">
</form>

<?php
if ($_SERVER["REQUEST_METHOD"] == "POST") {

    // Take user input
    $marks = $_POST['marks'];

    // if-else
    if ($marks >= 50) {
        echo "You passed!<br>";
    } else {
        echo "You failed!<br>";
    }

    echo "<br>Printing numbers 1 to 5:<br>";

    // loop
    for ($i = 1; $i <= 5; $i++) {
        echo "Number: $i <br>";
    }
}
?>

</body>
</html>
