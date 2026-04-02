param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $MessageParts
)

$Message = ($MessageParts -join ' ').Trim()

cargo +nightly fmt
git add .

if ($Message) {
    git commit -m $Message
} else {
    git commit
}