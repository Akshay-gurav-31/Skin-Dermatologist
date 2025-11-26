
 <!-- HTML Form -->
<form method="POST">
    Enter Number 1: <input type="number" name="num1"><br><br>
    Enter Number 2: <input type="number" name="num2"><br><br>
    <input type="submit" value="Calculate Sum">
</form>

<?php
// When form is submitted
if ($_SERVER["REQUEST_METHOD"] == "POST") {

    $num1 = $_POST['num1'];
    $num2 = $_POST['num2'];

    $sum = $num1 + $num2;

    echo "<h3>The sum of $num1 and $num2 is: $sum</h3>";
}
?>
