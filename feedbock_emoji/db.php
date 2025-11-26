<?php
// db.php - Database connection file


$DB_HOST = 'localhost';
$DB_USER = 'root';
$DB_PASS = 'OMKARm*@46!';
$DB_NAME = 'feedback_db';


try {
$pdo = new PDO("mysql:host=$DB_HOST;dbname=$DB_NAME;charset=utf8mb4", $DB_USER, $DB_PASS);
$pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
} catch (PDOException $e) {
die('Database connection failed: ' . $e->getMessage());
}
?>