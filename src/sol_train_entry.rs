//! This wrapper, or generally having the sol_train entry point be directly in `src`,
//! seems to be required for it to have access to other Molchanica code.

use molchanica::pharmacokinetics::sol_train;

fn main() {
    sol_train::main();
}
