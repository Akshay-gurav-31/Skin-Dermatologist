<?php
// dashboard.php
session_start([
    'cookie_httponly' => true,
    'cookie_samesite' => 'Lax'
]);

define('INACTIVE_TIMEOUT', 600);

// Check logged in and activity
if (empty($_SESSION['user'])) {
    header('Location: login.php');
    exit;
}

// Inactivity logout
if (time() - ($_SESSION['last_activity'] ?? 0) > INACTIVE_TIMEOUT) {
    // Timeout: destroy session and redirect to login with message
    session_unset();
    session_destroy();
    header('Location: login.php');
    exit;
}

// Update last activity time
$_SESSION['last_activity'] = time();

// Example protected content
$username = htmlspecialchars($_SESSION['user']);
?>
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Dashboard - Demo</title>
</head>
<body>
    <h2>Welcome, <?php echo $username; ?>!</h2>

    <p>This is a protected page that only logged-in users can see.</p>

    <p>
        <a href="logout.php">Logout</a>
    </p>

    <!-- Example: simple user details -->
    <h3>Session Info</h3>
    <ul>
        <li>User: <?php echo $username; ?></li>
        <li>Last activity: <?php echo date('Y-m-d H:i:s', $_SESSION['last_activity']); ?></li>
    </ul>
</body>
</html>
