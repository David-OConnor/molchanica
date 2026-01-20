//! This wrapper, or generally having the sol_train entry point be directly in `src`,
//! seems to be required for it to have access to other Molchanica code.

use molchanica::pharmacokinetics::train;

fn main() {
    sol_train::main();
}
