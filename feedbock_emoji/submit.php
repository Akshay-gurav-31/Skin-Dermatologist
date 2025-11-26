<?php
// submit.php - handles form submission
require 'db.php';

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $name = trim($_POST['name'] ?? '');
    $contact = trim($_POST['contact'] ?? '');
    $email = trim($_POST['email'] ?? '');
    $feedback = trim($_POST['feedback'] ?? '');
    $rating = intval($_POST['rating'] ?? 0);

    $errors = [];
    if ($name === '') $errors[] = 'Name is required.';
    if ($contact === '') $errors[] = 'Contact number is required.';
    if ($email === '' || !filter_var($email, FILTER_VALIDATE_EMAIL)) $errors[] = 'A valid email is required.';
    if ($feedback === '') $errors[] = 'Feedback cannot be empty.';
    if ($rating < 1 || $rating > 5) $errors[] = 'Please select a rating.';

    if (!empty($errors)) {
        echo "<h3>Errors:</h3><ul>";
        foreach ($errors as $err) echo "<li>" . htmlspecialchars($err) . "</li>";
        echo "</ul><p><a href='index.html'>Go Back</a></p>";
        exit;
    }

    try {
        $stmt = $pdo->prepare("INSERT INTO feedbacks (name, contact, email, feedback, rating, created_at) VALUES (:name, :contact, :email, :feedback, :rating, NOW())");
        $stmt->execute([
            ':name' => $name,
            ':contact' => $contact,
            ':email' => $email,
            ':feedback' => $feedback,
            ':rating' => $rating
        ]);

        header('Location: thankyou.php');
        exit;

    } catch (PDOException $e) {
        echo "Database error: " . $e->getMessage();
        exit;
    }
}
?>
