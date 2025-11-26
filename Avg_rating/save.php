<?php
require "db.php";

$r1 = $_POST['r1'];
$r2 = $_POST['r2'];
$r3 = $_POST['r3'];
$r4 = $_POST['r4'];
$r5 = $_POST['r5'];

$avg = ($r1 + $r2 + $r3 + $r4 + $r5) / 5;

$stmt = $pdo->prepare("
    INSERT INTO rating_sets (r1, r2, r3, r4, r5, average)
    VALUES (?, ?, ?, ?, ?, ?)
");
$stmt->execute([$r1, $r2, $r3, $r4, $r5, $avg]);

?>

<!DOCTYPE html>
<html>
<head>
    <title>Average Rating</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>

<div class="container">
    <h2>⭐ Calculated Average ⭐</h2>
    <p style="text-align:center; font-size:22px; font-weight:bold;">
        Average Rating: <?= round($avg,2) ?> ⭐
    </p>
    <a href="index.html">
        <button>Back</button>
    </a>
</div>

</body>
</html>
