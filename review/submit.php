<?php
include "db.php";

$name = $_POST['name'];
$email = $_POST['email'];
$rating = $_POST['rating'];
$review = $_POST['review'];

$sql = "INSERT INTO reviews (name, email, rating, review) 
        VALUES ('$name', '$email', '$rating', '$review')";

if(mysqli_query($conn, $sql)){
    echo "<h2>Thank you! Your review has been submitted.</h2>";
} else {
    echo "Error: " . mysqli_error($conn);
}
?>
	