use std::mem;

use crate::opcode::Opcode;
use crate::region::Region;

/// A the type of a single register in our virtual machine
pub type Register = isize;

#[derive(Debug, Clone, Copy, PartialEq)]
enum State {
  Active,
  Halted,
}

/// A virtual machine for SM213 architecture, with slight modifications.
///
/// In the original spec, it was noted that:
///
/// "Offsets indicated with o in assembly are stored in compressed form as p in
/// machine code (meaning o × 2 or o × 4, as in the semantics column"
///
/// We instead just use the word size of the target platform...
#[derive(Debug)]
pub struct Vm {
  // this represents the current instruction nibble we are on
  ip: usize,
  memory: Vec<isize>,
  registers: [Register; 7],
  state: State,
}

const MAX_MEMORY: usize = 32;

impl Vm {
  /// Create a new, empty virtual machine
  pub fn new() -> Self {
    Self {
      ip: 0,
      memory: vec![0; MAX_MEMORY],
      registers: [0; 7],
      state: State::Active,
    }
  }

  /// Step through a single opcode/instruction, indicating failure
  pub fn step<R>(&mut self, region: &R) -> Option<()>
  where
    R: Region,
  {
    if self.state == State::Halted {
      return None;
    }
    let mut task = Task::new(self, region);
    task.run()?;
    Some(())
  }

  fn read_word(&self, address: usize) -> Option<isize> {
    self.memory.get(address).copied()
  }

  fn write_word(&mut self, address: usize, value: isize) -> Option<()> {
    self.memory.get_mut(address).map(|prev| {
      *prev = value;
    })
  }
}

impl Default for Vm {
  fn default() -> Self {
    Self::new()
  }
}

/// An error that occurred during execution of instructions
///
/// TODO: replace all options with result, and make an error variant for each
/// unique point of failure...
#[derive(thiserror::Error, Debug)]
pub enum Error {
  #[error("did not expect to reach the end of instructions")]
  UnexpectedEof,

  #[error("machine is halted")]
  MachineHalted,
}

struct Task<'vm, 'region, R> {
  vm: &'vm mut Vm,
  region: &'region R,
}

impl<'vm, 'region, R> Task<'vm, 'region, R>
where
  R: Region,
{
  fn new(vm: &'vm mut Vm, region: &'region R) -> Self {
    Self { vm, region }
  }

  #[inline]
  fn eat(&mut self) -> Option<u8> {
    let byte = self.region.instructions().get(self.vm.ip / 2)?;
    let parity = self.vm.ip & 0x1;
    // we want to shift the byte down if its even (the first nibble),
    // otherwise we want the lowest anyway...
    let nibble = (byte >> (4 * (1 - parity))) & 0xF;
    self.vm.ip += 1;
    Some(nibble)
  }

  fn eat_immediate(&mut self) -> Option<Register> {
    let mut result = 0;
    for _ in 0..8 {
      result = (result << 4) | self.eat()? as isize;
    }
    Some(result)
  }

  fn run(&mut self) -> Option<()> {
    let pc = self.vm.ip;
    let op: Opcode = self.eat()?.into();
    match op {
      Opcode::LoadImmediate => load_immediate(self)?,
      Opcode::LoadBaseOff => load_base_off(self)?,
      Opcode::LoadIndexed => load_indexed(self)?,
      Opcode::StoreBaseOff => store_base_off(self)?,
      Opcode::StoreIndexed => store_indexed(self)?,
      Opcode::Miscellaneous => miscellaneous(self, pc)?,
      Opcode::Shift => shift(self)?,
      Opcode::Branch => branch(self, pc)?,
      Opcode::BranchIfEqual => branch_if_equal(self, pc)?,
      Opcode::BranchIfGreater => branch_if_greater(self, pc)?,
      Opcode::JumpImmediate => jump_immediate(self)?,
      Opcode::JumpBaseOff => jump_base_off(self)?,
      Opcode::JumpIndirBaseOff => jump_indir_base_off(self)?,
      Opcode::JumpIndirIndex => jump_indir_index(self)?,
      Opcode::Nop => nop(self)?,
    }
    Some(())
  }
}

// r[d] ← vvvvvvvv
fn load_immediate<R>(task: &mut Task<'_, '_, R>) -> Option<()>
where
  R: Region,
{
  let d = task.eat()? as usize;
  let _ = task.eat()?;
  let _ = task.eat()?;
  let vs = task.eat_immediate()?;
  task.vm.registers[d] = vs;
  Some(())
}

// r[d] ← m[(o = p × word_size) + r[s]]
fn load_base_off<R>(task: &mut Task<'_, '_, R>) -> Option<()>
where
  R: Region,
{
  let p = task.eat()? as usize;
  let s = task.eat()? as usize;
  let d = task.eat()? as usize;
  // i actually change the spec a bit here to match variable word size...
  let o = p * mem::size_of::<isize>(); // is now: o = p × sizeof(uintptr_t)
  let rs: usize = task.vm.registers[s].try_into().ok()?; // r[s]
  let target = o + rs;
  let value = task.vm.read_word(target)?;
  task.vm.registers[d] = value;
  Some(())
}

// r[d] ← m[r[s] + r[i] × word_size]
fn load_indexed<R>(task: &mut Task<'_, '_, R>) -> Option<()>
where
  R: Region,
{
  let s = task.eat()? as usize;
  let i = task.eat()? as usize;
  let d = task.eat()? as usize;
  let rs: usize = task.vm.registers[s].try_into().ok()?;
  let ri: usize = task.vm.registers[i].try_into().ok()?;
  let target = rs + ri * mem::size_of::<isize>();
  let value = task.vm.read_word(target)?;
  task.vm.registers[d] = value;
  Some(())
}

// m[(o = p × word_size) + r[d]] ← r[s]
fn store_base_off<R>(task: &mut Task<'_, '_, R>) -> Option<()>
where
  R: Region,
{
  let s = task.eat()? as usize;
  let p = task.eat()? as usize;
  let d = task.eat()? as usize;
  let o = p * mem::size_of::<isize>();
  let rd: usize = task.vm.registers[d].try_into().ok()?; // r[d]
  let target = o + rd;
  let value = task.vm.registers[s]; // r[s]
  task.vm.write_word(target, value)?;
  Some(())
}

// m[r[d] + r[i] × word_size] ← r[s]
fn store_indexed<R>(task: &mut Task<'_, '_, R>) -> Option<()>
where
  R: Region,
{
  let s = task.eat()? as usize;
  let d = task.eat()? as usize;
  let i = task.eat()? as usize;
  let rd: usize = task.vm.registers[d].try_into().ok()?; // r[d]
  let ri: usize = task.vm.registers[i].try_into().ok()?; // r[i]
  let target = rd + ri * mem::size_of::<isize>();
  let value = task.vm.registers[s]; // r[s]
  task.vm.write_word(target, value)?;
  Some(())
}

fn miscellaneous<R>(task: &mut Task<'_, '_, R>, pc: usize) -> Option<()>
where
  R: Region,
{
  let subcode = task.eat()?;
  match subcode {
    0x0 => {
      // r[d] ← r[s]
      let s = task.eat()? as usize;
      let d = task.eat()? as usize;
      task.vm.registers[d] = task.vm.registers[s];
    }
    0x1 => {
      // r[d] ← r[d] + r[s]
      let s = task.eat()? as usize;
      let d = task.eat()? as usize;
      task.vm.registers[d] += task.vm.registers[s];
    }
    0x2 => {
      // r[d] ← r[d] & r[s]
      let s = task.eat()? as usize;
      let d = task.eat()? as usize;
      task.vm.registers[d] &= task.vm.registers[s];
    }
    0x3 => {
      // r[d] ← r[d] + 1
      let _ = task.eat()?;
      let d = task.eat()? as usize;
      task.vm.registers[d] += 1;
    }
    0x4 => {
      // r[d] ← r[d] + word_size
      let _ = task.eat()?;
      let d = task.eat()? as usize;
      task.vm.registers[d] += mem::size_of::<isize>() as isize;
    }
    0x5 => {
      // r[d] ← r[d] − 1
      let _ = task.eat()?;
      let d = task.eat()? as usize;
      task.vm.registers[d] -= 1;
    }
    0x6 => {
      // r[d] ← r[d] - word_size
      let _ = task.eat()?;
      let d = task.eat()? as usize;
      task.vm.registers[d] -= mem::size_of::<isize>() as isize;
    }
    0x7 => {
      // r[d] ← ~r[d]
      let _ = task.eat()?;
      let d = task.eat()? as usize;
      task.vm.registers[d] = !task.vm.registers[d];
    }
    0xF => {
      // r[d] ← pc + (o = 2 × p)
      let p = task.eat()? as usize;
      let d = task.eat()? as usize;
      let o = 2 * p;
      task.vm.registers[d] = (pc + o) as isize;
    }
    _ => panic!("impossible subcode for misc"),
  }
  Some(())
}

// r[d] ← r[d] << vv
// r[d] ← r[d] >> −vv (if vv is negative)
fn shift<R>(task: &mut Task<'_, '_, R>) -> Option<()>
where
  R: Region,
{
  let d = task.eat()? as usize;
  let vv = ((task.eat()? << 4) | task.eat()?) as i8;
  if vv >= 0 {
    task.vm.registers[d] <<= vv;
  } else {
    task.vm.registers[d] >>= -vv;
  }
  Some(())
}

// pc ← (aaaaaaaa = pc + pp × 2)
fn branch<R>(task: &mut Task<'_, '_, R>, pc: usize) -> Option<()>
where
  R: Region,
{
  let _ = task.eat()?;
  let pp = ((task.eat()? << 4) | task.eat()?) as i8 as isize;
  let r#as = pc as isize + pp * 2;
  task.vm.ip = r#as as usize;
  Some(())
}

// if r[s] == 0 : pc ← (aaaaaaaa = pc + pp × 2)
fn branch_if_equal<R>(task: &mut Task<'_, '_, R>, pc: usize) -> Option<()>
where
  R: Region,
{
  let s = task.eat()? as usize;
  let pp = ((task.eat()? << 4) | task.eat()?) as i8 as isize;
  let r#as = pc as isize + pp * 2;
  if task.vm.registers[s] == 0 {
    task.vm.ip = r#as as usize;
  }
  Some(())
}

// if r[s] > 0 : pc ← (aaaaaaaa = pc + pp × 2)
fn branch_if_greater<R>(task: &mut Task<'_, '_, R>, pc: usize) -> Option<()>
where
  R: Region,
{
  let s = task.eat()? as usize;
  let pp = ((task.eat()? << 4) | task.eat()?) as i8 as isize;
  let r#as = pc as isize + pp * 2;
  if task.vm.registers[s] > 0 {
    task.vm.ip = r#as as usize;
  }
  Some(())
}

// pc ← aaaaaaaa with .pos aaaaaaaa label:
fn jump_immediate<R>(task: &mut Task<'_, '_, R>) -> Option<()>
where
  R: Region,
{
  let _ = task.eat()?;
  let _ = task.eat()?;
  let _ = task.eat()?;
  let r#as: usize = task.eat_immediate()?.try_into().ok()?;
  task.vm.ip = r#as;
  Some(())
}

// pc ← r[s] + (o = 2 × pp)
fn jump_base_off<R>(task: &mut Task<'_, '_, R>) -> Option<()>
where
  R: Region,
{
  let s = task.eat()? as usize;
  let pp = ((task.eat()? << 4) | task.eat()?) as i8 as isize;
  let rs = task.vm.registers[s];
  let o = 2 * pp;
  let target = rs + o;
  task.vm.ip = target.try_into().ok()?;
  Some(())
}

// pc ← m[(o = word_size × pp) + r[s]]
fn jump_indir_base_off<R>(task: &mut Task<'_, '_, R>) -> Option<()>
where
  R: Region,
{
  let s = task.eat()? as usize;
  let pp = ((task.eat()? << 4) | task.eat()?) as i8 as isize;
  let rs = task.vm.registers[s];
  let o = mem::size_of::<isize>() as isize * pp;
  let target = o + rs;
  let target: usize = target.try_into().ok()?;
  task.vm.ip = task.vm.read_word(target)?.try_into().ok()?;
  Some(())
}

// pc ← m[word_size × r[i] + r[s]]
fn jump_indir_index<R>(task: &mut Task<'_, '_, R>) -> Option<()>
where
  R: Region,
{
  let s = task.eat()? as usize;
  let i = task.eat()? as usize;
  let _ = task.eat()?;
  let rs = task.vm.registers[s];
  let ri = task.vm.registers[i];
  let target = mem::size_of::<isize>() as isize * ri + rs;
  let target: usize = target.try_into().ok()?;
  task.vm.ip = task.vm.read_word(target)?.try_into().ok()?;
  Some(())
}

// (stop execution) ∨ (do nothing)
fn nop<R>(task: &mut Task<'_, '_, R>) -> Option<()>
where
  R: Region,
{
  let flag = task.eat()?;
  if flag == 0x0 {
    task.vm.state = State::Halted;
  } else {
    // do nuffin
  }
  let _ = task.eat()?;
  let _ = task.eat()?;
  Some(())
}

#[cfg(test)]
mod tests {
  use super::*;

  fn all_zeroed(iter: impl Iterator<Item = isize>) {
    for i in iter {
      assert_eq!(i, 0);
    }
  }

  mod vm {
    use super::*;

    use crate::region::Chunk;

    #[test]
    fn new() {
      let vm = Vm::new();
      assert_eq!(vm.ip, 0);
      all_zeroed(vm.registers.into_iter());
      all_zeroed(vm.memory.into_iter());
    }

    #[test]
    fn step_load_immediate() {
      let chunk: Chunk = vec![0x01, 0x00, 0x00, 0x00, 0x10, 0x00].into();
      let mut vm = Vm::new();
      assert_eq!(vm.step(&chunk), Some(()));
      assert_eq!(vm.registers[1], 0x1000);
    }

    #[test]
    fn step_load_base_off() {
      let chunk: Chunk = vec![0x11, 0x23].into();
      let mut vm = Vm::new();
      vm.registers[2] = 1; // r[s] = 1
      vm.registers[3] = 0; // r[d] = 0
      vm.memory[9] = 42; // 9 = (8 * 1) + 1
      assert_eq!(vm.step(&chunk), Some(()));
      assert_eq!(vm.registers[3], 42); // r[d] = 42
      assert_eq!(vm.memory[9], 42);
    }

    #[test]
    fn step_load_indexed() {
      let chunk: Chunk = vec![0x22, 0x34].into();
      let mut vm = Vm::new();
      vm.registers[2] = 3; // r[s] = 8
      vm.registers[3] = 1; // r[i] = 2
      vm.registers[4] = 0; // r[d] = 0
      let word_size = mem::size_of::<isize>();
      let effective_address = 3 + 1 * word_size;
      vm.memory[effective_address] = 42;
      assert_eq!(vm.step(&chunk), Some(()));
      assert_eq!(vm.registers[4], 42);
      assert_eq!(vm.memory[effective_address], 42);
    }

    #[test]
    fn step_store_base_off() {
      let chunk: Chunk = vec![0x32, 0x13].into();
      let mut vm = Vm::new();
      vm.registers[2] = 42; // r[s] = 42
      vm.registers[3] = 1; // r[d] = 1
      let word_size = mem::size_of::<isize>();
      let effective_address = 1 + 1 * word_size;
      assert_eq!(vm.step(&chunk), Some(()));
      assert_eq!(vm.memory[effective_address], 42);
    }

    #[test]
    fn step_store_indexed() {
      let chunk: Chunk = vec![0x42, 0x34].into();
      let mut vm = Vm::new();
      vm.registers[2] = 42; // r[s] = 42
      vm.registers[3] = 1; // r[d] = 1
      vm.registers[4] = 2; // r[i] = 2
      let word_size = mem::size_of::<isize>();
      let effective_address = 1 + 2 * word_size;
      assert_eq!(vm.step(&chunk), Some(()));
      assert_eq!(vm.memory[effective_address], 42);
    }

    #[test]
    fn step_misc_rr_move() {
      let chunk: Chunk = vec![0x60, 0x12].into(); // s=1, d=2
      let mut vm = Vm::new();
      vm.registers[1] = 42; // r[s] = 42
      assert_eq!(vm.step(&chunk), Some(()));
      assert_eq!(vm.registers[2], 42); // r[d] = 42
    }

    #[test]
    fn step_misc_add() {
      let chunk: Chunk = vec![0x61, 0x12].into(); // s=1, d=2
      let mut vm = Vm::new();
      vm.registers[1] = 10; // r[s] = 10
      vm.registers[2] = 20; // r[d] = 20
      assert_eq!(vm.step(&chunk), Some(()));
      assert_eq!(vm.registers[2], 30); // r[d] = 30
    }

    #[test]
    fn step_misc_and() {
      let chunk: Chunk = vec![0x62, 0x12].into(); // s=1, d=2
      let mut vm = Vm::new();
      vm.registers[1] = 0b1100; // r[s] = 12
      vm.registers[2] = 0b1010; // r[d] = 10
      assert_eq!(vm.step(&chunk), Some(()));
      assert_eq!(vm.registers[2], 0b1000); // r[d] = 8
    }

    #[test]
    fn step_misc_inc() {
      let chunk: Chunk = vec![0x63, 0x01].into(); // d=1
      let mut vm = Vm::new();
      vm.registers[1] = 10; // r[d] = 10
      assert_eq!(vm.step(&chunk), Some(()));
      assert_eq!(vm.registers[1], 11); // r[d] = 11
    }

    #[test]
    fn step_misc_inc_addr() {
      let chunk: Chunk = vec![0x64, 0x01].into(); // d=1
      let mut vm = Vm::new();
      vm.registers[1] = 8; // r[d] = 8
      let word_size = mem::size_of::<isize>() as isize;
      assert_eq!(vm.step(&chunk), Some(()));
      assert_eq!(vm.registers[1], 8 + word_size); // r[d] = 8 + word_size
    }

    #[test]
    fn step_misc_dec() {
      let chunk: Chunk = vec![0x65, 0x01].into(); // d=1
      let mut vm = Vm::new();
      vm.registers[1] = 10; // r[d] = 10
      assert_eq!(vm.step(&chunk), Some(()));
      assert_eq!(vm.registers[1], 9); // r[d] = 9
    }

    #[test]
    fn step_misc_dec_addr() {
      let chunk: Chunk = vec![0x66, 0x01].into(); // d=1
      let mut vm = Vm::new();
      vm.registers[1] = 16; // r[d] = 16
      let word_size = mem::size_of::<isize>() as isize;
      assert_eq!(vm.step(&chunk), Some(()));
      assert_eq!(vm.registers[1], 16 - word_size); // r[d] = 16 - word_size
    }

    #[test]
    fn step_misc_not() {
      let chunk: Chunk = vec![0x67, 0x01].into(); // d=1
      let mut vm = Vm::new();
      vm.registers[1] = 0b1010; // r[d] = 10
      assert_eq!(vm.step(&chunk), Some(()));
      assert_eq!(vm.registers[1], !0b1010); // r[d] = ~10
    }

    #[test]
    fn step_misc_get_pc_start() {
      let chunk: Chunk = vec![0x6F, 0x31].into();
      let mut vm = Vm::new();
      assert_eq!(vm.step(&chunk), Some(()));
      assert_eq!(vm.registers[1], 6); // r1 = 0 + (2 * 3) = 0 + 6 = 6
    }

    #[test]
    fn step_misc_get_pc_end() {
      #[rustfmt::skip]
      let chunk: Chunk = vec![
        0x01, 0x00, 0x00, 0x00, 0x10, 0x00,
        0x02, 0x00, 0x00, 0x00, 0x20, 0x00,
        0x6F, 0x31,
      ]
      .into();
      let mut vm = Vm::new();
      assert_eq!(vm.step(&chunk), Some(()));
      assert_eq!(vm.registers[1], 0x1000);
      assert_eq!(vm.step(&chunk), Some(()));
      assert_eq!(vm.registers[2], 0x2000);
      assert_eq!(vm.step(&chunk), Some(()));
      assert_eq!(vm.registers[1], 30); // r1 = 24 + (2 * 3) = 24 + 6 = 30
    }

    #[test]
    fn step_shift_left() {
      let chunk: Chunk = vec![0x71, 0x02].into();
      let mut vm = Vm::new();
      vm.registers[1] = 0b0001; // r1 = 1
      assert_eq!(vm.step(&chunk), Some(()));
      assert_eq!(vm.registers[1], 0b0100); // r1 <<= 1 becomes 4
    }

    #[test]
    fn step_shift_right() {
      // -2 = 0b0000 0010 -> 0b1111 1101 -> 0b1111 1110 -> 0xFE
      let chunk: Chunk = vec![0x71, 0xFE].into();
      let mut vm = Vm::new();
      vm.registers[1] = 0b1000; // r1 = 1
      assert_eq!(vm.step(&chunk), Some(()));
      assert_eq!(vm.registers[1], 0b0010); // r1 >>= 8 becomes 2
    }

    #[test]
    fn step_branch_forward() {
      let chunk: Chunk = vec![0x80, 0x02].into();
      let mut vm = Vm::new();
      vm.ip = 0; // pc = 0
      assert_eq!(vm.step(&chunk), Some(()));
      assert_eq!(vm.ip, 4); // pc = 0 + (2 * 2) = 4
    }

    #[test]
    fn step_branch_backward() {
      #[rustfmt::skip]
      let chunk: Chunk = vec![
        0x01, 0x00, 0x00, 0x00, 0x10, 0x00,
        0x02, 0x00, 0x00, 0x00, 0x20, 0x00,
        0x8F, 0xFA,
      ]
      .into();
      let mut vm = Vm::new();
      // skip two
      assert_eq!(vm.step(&chunk), Some(()));
      assert_eq!(vm.step(&chunk), Some(()));
      // now we branch
      assert_eq!(vm.ip, 24);
      assert_eq!(vm.step(&chunk), Some(()));
      assert_eq!(vm.ip, 12); // pc = 24 + (-6 * 2) = 20
    }

    #[test]
    fn step_branch_if_equal_zero() {
      #[rustfmt::skip]
      let chunk: Chunk = vec![
        0x01, 0x00, 0x00, 0x00, 0x10, 0x00,
        0x02, 0x00, 0x00, 0x00, 0x20, 0x00,
        0x91, 0xFA,
      ]
      .into();
      let mut vm = Vm::new();
      assert_eq!(vm.step(&chunk), Some(()));
      assert_eq!(vm.step(&chunk), Some(()));
      vm.registers[1] = 0; // meet branch condition
      assert_eq!(vm.ip, 24);
      assert_eq!(vm.step(&chunk), Some(()));
      assert_eq!(vm.ip, 12); // pc = 24 + (-6 * 2) = 12
    }

    #[test]
    fn step_branch_if_equal_nonzero() {
      #[rustfmt::skip]
      let chunk: Chunk = vec![
        0x01, 0x00, 0x00, 0x00, 0x10, 0x00,
        0x02, 0x00, 0x00, 0x00, 0x20, 0x00,
        0x91, 0xFA,
      ]
      .into();
      let mut vm = Vm::new();
      assert_eq!(vm.step(&chunk), Some(()));
      assert_eq!(vm.step(&chunk), Some(()));
      vm.registers[1] = 1; // do not meet branch condition
      assert_eq!(vm.ip, 24);
      assert_eq!(vm.step(&chunk), Some(()));
      assert_eq!(vm.ip, 28); // noop, we should have moved past
    }

    #[test]
    fn step_branch_if_greater_zero() {
      #[rustfmt::skip]
      let chunk: Chunk = vec![
        0x01, 0x00, 0x00, 0x00, 0x10, 0x00,
        0x02, 0x00, 0x00, 0x00, 0x20, 0x00,
        0xA1, 0xFA,
      ]
      .into();
      let mut vm = Vm::new();
      assert_eq!(vm.step(&chunk), Some(()));
      assert_eq!(vm.step(&chunk), Some(()));
      vm.registers[1] = 0; // do not meet branch condition
      assert_eq!(vm.ip, 24);
      assert_eq!(vm.step(&chunk), Some(()));
      assert_eq!(vm.ip, 28); // noop, we should have moved past
    }

    #[test]
    fn step_branch_if_greater_nonzero() {
      #[rustfmt::skip]
      let chunk: Chunk = vec![
        0x01, 0x00, 0x00, 0x00, 0x10, 0x00,
        0x02, 0x00, 0x00, 0x00, 0x20, 0x00,
        0xA1, 0xFA,
      ]
      .into();
      let mut vm = Vm::new();
      assert_eq!(vm.step(&chunk), Some(()));
      assert_eq!(vm.step(&chunk), Some(()));
      vm.registers[1] = 1; // meet branch condition
      assert_eq!(vm.ip, 24);
      assert_eq!(vm.step(&chunk), Some(()));
      assert_eq!(vm.ip, 12); // pc = 24 + (-6 * 2) = 12
    }

    #[test]
    fn step_jump_immediate() {
      #[rustfmt::skip]
      let chunk: Chunk = vec![
        0x01, 0x00, 0x00, 0x00, 0x10, 0x00,
        0x02, 0x00, 0x00, 0x00, 0x20, 0x00,
        0xB0, 0x00, 0x00, 0x00, 0x00, 0x0C,
      ]
      .into();
      let mut vm = Vm::new();
      assert_eq!(vm.step(&chunk), Some(()));
      assert_eq!(vm.step(&chunk), Some(()));
      assert_eq!(vm.ip, 24);
      assert_eq!(vm.step(&chunk), Some(()));
      assert_eq!(vm.ip, 12);
    }

    #[test]
    fn step_jump_base_off() {
      #[rustfmt::skip]
      let chunk: Chunk = vec![
        0x01, 0x00, 0x00, 0x00, 0x00, 0x0A, // we store 10 here
        0x02, 0x00, 0x00, 0x00, 0x20, 0x00,
        0xC1, 0x01,
      ]
      .into();
      let mut vm = Vm::new();
      assert_eq!(vm.step(&chunk), Some(()));
      assert_eq!(vm.step(&chunk), Some(()));
      assert_eq!(vm.ip, 24);
      assert_eq!(vm.step(&chunk), Some(()));
      assert_eq!(vm.ip, 12); // pc = r1 + (1 × 2) = 10 + 2 = 12
    }

    #[test]
    fn step_jump_indir_base_off() {
      #[rustfmt::skip]
      let chunk: Chunk = vec![
        0x01, 0x00, 0x00, 0x00, 0x00, 0x0A, // we store 10 here
        0x02, 0x00, 0x00, 0x00, 0x20, 0x00,
        0xD1, 0x01,
      ]
      .into();
      let mut vm = Vm::new();
      vm.memory[0xA + mem::size_of::<isize>() * 1] = 12;
      assert_eq!(vm.step(&chunk), Some(()));
      assert_eq!(vm.step(&chunk), Some(()));
      assert_eq!(vm.ip, 24);
      assert_eq!(vm.step(&chunk), Some(()));
      assert_eq!(vm.ip, 12);
    }
    
    //pc ← m[word_size × r[i] + r[s]]
    #[test]
    fn step_jump_indir_index() {
      #[rustfmt::skip]
      let chunk: Chunk = vec![
        0x01, 0x00, 0x00, 0x00, 0x00, 0x04, // store 4 (r[s])
        0x02, 0x00, 0x00, 0x00, 0x00, 0x01, // store 1 (r[i])
        0xE1, 0x20,
      ]
      .into();
      let mut vm = Vm::new();
      vm.memory[0x4 + mem::size_of::<isize>() * 1] = 12;
      assert_eq!(vm.step(&chunk), Some(()));
      assert_eq!(vm.step(&chunk), Some(()));
      assert_eq!(vm.ip, 24);
      assert_eq!(vm.step(&chunk), Some(()));
      assert_eq!(vm.ip, 12);
    }

    #[test]
    fn step_nop() {
      let chunk: Chunk = vec![0xFF, 0x00].into();
      let mut vm = Vm::new();
      assert_eq!(vm.step(&chunk), Some(()));
      assert_eq!(vm.ip, 4);
      assert_eq!(vm.state, State::Active);
      all_zeroed(vm.memory.into_iter());
      all_zeroed(vm.registers.into_iter());
    }

    #[test]
    fn step_nop_halt() {
      let chunk: Chunk = vec![0xFF, 0x00, 0xF0, 0x00].into();
      let mut vm = Vm::new();
      // normal nop
      assert_eq!(vm.step(&chunk), Some(()));
      assert_eq!(vm.ip, 4);
      assert_eq!(vm.state, State::Active);
      // execute halt
      assert_eq!(vm.step(&chunk), Some(()));
      assert_eq!(vm.ip, 8);
      assert_eq!(vm.state, State::Halted);
      // cant progress
      assert_eq!(vm.step(&chunk), None);
      assert_eq!(vm.ip, 8);
    }
  }
}
