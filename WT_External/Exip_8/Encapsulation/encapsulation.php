
<?php
class BankAccount {
    private $balance;

    public function setBalance($amount) {
        $this->balance = $amount;
    }

    public function getBalance() {
        return $this->balance;
    }
}

$b = new BankAccount();
$b->setBalance(5000);
echo $b->getBalance();
?>
