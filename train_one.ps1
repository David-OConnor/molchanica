param(
    [Parameter(Mandatory = $true)]
    [string]$args0
)

cargo r --release --features train --bin train -- --path C:/Users/the_a/Desktop/bio_misc/tdc_data --tgt $args0 --eval