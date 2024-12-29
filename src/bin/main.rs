use emulator::region::Chunk;
use emulator::vm::Vm;

fn main() {
  let chunk: Chunk = vec![0x01, 0x00, 0x00, 0x00, 0x10, 0x00].into();
  let mut vm = Vm::new();
  dbg!(vm.step(&chunk));
  dbg!(vm);
}
