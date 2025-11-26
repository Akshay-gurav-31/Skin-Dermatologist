<?php
// login.php
// Start session with secure cookie flags
session_start([
    'cookie_httponly' => true,
    'cookie_samesite' => 'Lax'
]);

// Inactivity timeout in seconds (e.g., 10 minutes)
define('INACTIVE_TIMEOUT', 600);

// If user already logged in and active, redirect to dashboard
if (isset($_SESSION['user']) && (time() - ($_SESSION['last_activity'] ?? 0) <= INACTIVE_TIMEOUT)) {
    header('Location: dashboard.php');
    exit;
}

// Generate CSRF token for the form (single-use token)
if (empty($_SESSION['csrf_token'])) {
    $_SESSION['csrf_token'] = bin2hex(random_bytes(16));
}

// Hardcoded users (example). In real apps use a DB.
$users = [
    // username => password_hash('plainpassword', PASSWORD_DEFAULT)
    'akash' => password_hash('pass123', PASSWORD_DEFAULT),
    'alice' => password_hash('alicepwd', PASSWORD_DEFAULT)
];

$errors = [];

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    // Basic CSRF check
    $token = $_POST['csrf_token'] ?? '';
    if (empty($token) || !hash_equals($_SESSION['csrf_token'], $token)) {
        $errors[] = "Invalid request (CSRF token mismatch).";
    } else {
        // Get inputs (trim)
        $username = trim($_POST['username'] ?? '');
        $password = $_POST['password'] ?? '';

        // Basic validation
        if ($username === '' || $password === '') {
            $errors[] = "Enter both username and password.";
        } else {
            // Authenticate: check user exists and password matches
            if (isset($users[$username]) && password_verify($password, $users[$username])) {
                // Successful login
                session_regenerate_id(true); // prevent session fixation
                $_SESSION['user'] = $username;
                $_SESSION['last_activity'] = time();

                // Regenerate CSRF token for future forms
                $_SESSION['csrf_token'] = bin2hex(random_bytes(16));

                header('Location: dashboard.php');
                exit;
            } else {
                $errors[] = "Invalid username or password.";
            }
        }
    }
}
?>
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Login - Demo</title>
</head>
<body>
    <h2>Login</h2>

    <?php if (!empty($errors)): ?>
        <div style="color:red;">
            <?php foreach ($errors as $e) echo htmlspecialchars($e) . "<br>"; ?>
        </div>
    <?php endif; ?>

    <form method="post" action="login.php" autocomplete="off">
        <label>Username:
            <input type="text" name="username" required>
        </label><br><br>

        <label>Password:
            <input type="password" name="password" required>
        </label><br><br>

        <!-- CSRF token -->
        <input type="hidden" name="csrf_token" value="<?php echo htmlspecialchars($_SESSION['csrf_token']); ?>">

        <button type="submit">Login</button>
    </form>

    <p>Try users: <strong>akash / pass123</strong> or <strong>alice / alicepwd</strong></p>
</body>
</html>
