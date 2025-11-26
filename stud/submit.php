<?php
include "db.php";

$name = $_POST['name'];
$s1 = $_POST['s1'];
$s2 = $_POST['s2'];
$s3 = $_POST['s3'];
$s4 = $_POST['s4'];
$s5 = $_POST['s5'];

$sql = "INSERT INTO marks (name, subject1, subject2, subject3, subject4, subject5)
        VALUES ('$name', '$s1', '$s2', '$s3', '$s4', '$s5')";

if (mysqli_query($conn, $sql)) {
    echo "<h2>Marks Added Successfully</h2>";
    echo "<a href='average.php'>Click here to see Average Percentage</a>";
} else {
    echo "Error: " . mysqli_error($conn);
}

?>
