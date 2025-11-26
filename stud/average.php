<?php
include "db.php";

$sql = "SELECT (subject1 + subject2 + subject3 + subject4 + subject5) / 5 AS percentage FROM marks";
$result = mysqli_query($conn, $sql);

$totalPercentage = 0;
$count = 0;

while ($row = mysqli_fetch_assoc($result)) {
    $totalPercentage += $row['percentage'];
    $count++;
}

if ($count > 0) {
    $average = $totalPercentage / $count;
    echo "<h2>📊 Overall Average Percentage of All Students: " . round($average, 2) . "%</h2>";
} else {
    echo "<h3>No records found.</h3>";
}
?>
