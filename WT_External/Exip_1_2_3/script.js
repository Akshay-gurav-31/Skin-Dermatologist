// Registration validation
function validateRegister() {
    let name = document.getElementById("name").value;
    let email = document.getElementById("email").value;
    let pass = document.getElementById("password").value;
    let cpass = document.getElementById("confirm").value;
    let msg = document.getElementById("regMsg");

    if (name === "" || email === "" || pass === "") {
        msg.style.color = "red";
        msg.innerHTML = "All fields are required!";
        return false;
    }

    let emailCheck = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailCheck.test(email)) {
        msg.style.color = "red";
        msg.innerHTML = "Enter a valid email!";
        return false;
    }

    if (pass.length < 6) {
        msg.style.color = "red";
        msg.innerHTML = "Password must be at least 6 characters!";
        return false;
    }

    if (pass !== cpass) {
        msg.style.color = "red";
        msg.innerHTML = "Passwords do not match!";
        return false;
    }

    msg.style.color = "green";
    msg.innerHTML = "Registration Successful!";
    return false; // prevent form submit for demo
}



// Login validation
function validateLogin() {
    let email = document.getElementById("loginEmail").value;
    let pass = document.getElementById("loginPassword").value;
    let msg = document.getElementById("loginMsg");

    if (email === "" || pass === "") {
        msg.style.color = "red";
        msg.innerHTML = "Email and Password required!";
        return false;
    }

    let emailCheck = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailCheck.test(email)) {
        msg.style.color = "red";
        msg.innerHTML = "Enter a valid email!";
        return false;
    }

    if (pass.length < 6) {
        msg.style.color = "red";
        msg.innerHTML = "Invalid password!";
        return false;
    }

    msg.style.color = "green";
    msg.innerHTML = "Login Successful! Redirecting...";

    // Redirect to About Us page
    setTimeout(() => {
        window.location.href = "about.html";
    }, 800);

    return false;
}
