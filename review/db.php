<?php
$host = "localhost";
$user = "root";
$pass = "OMKARm*@46!";
$db = "review_db";

$conn = mysqli_connect($host, $user, $pass, $db);

if(!$conn){
    die("Database connection failed: " . mysqli_connect_error());
}
?>
