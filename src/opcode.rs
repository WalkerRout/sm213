#[repr(u8)]
#[derive(Debug, Clone, Copy)]
pub enum Opcode {
  /// Loads an immediate value into a register.
  ///
  /// | Operation      | Semantics/RTL     | Assembly           |
  /// |----------------|-------------------|--------------------|
  /// | Load Immediate | `r[d] ← vvvvvvvv` | `ld $vvvvvvvv, rd` |
  LoadImmediate = 0x0,

  /// Loads a value from memory using a base address and offset.
  ///
  /// | Operation        | Semantics/RTL               | Assembly            |
  /// |------------------|-----------------------------|---------------------|
  /// | Load Base+Offset | `r[d] ← Mem[r[b] + offset]` | `ld offset(rb), rd` |
  LoadBaseOff = 0x1,

  /// Loads a value from memory using an indexed address.
  ///
  /// | Operation    | Semantics/RTL             | Assembly          |
  /// |--------------|---------------------------|-------------------|
  /// | Load Indexed | `r[d] ← Mem[r[b] + r[x]]` | `ld (rb, rx), rd` |
  LoadIndexed = 0x2,

  /// Stores a value in memory using a base address and offset.
  ///
  /// | Operation         | Semantics/RTL               | Assembly            |
  /// |-------------------|-----------------------------|---------------------|
  /// | Store Base+Offset | `Mem[r[b] + offset] ← r[d]` | `st rd, offset(rb)` |
  StoreBaseOff = 0x3,

  /// Stores a value in memory using an indexed address.
  ///
  /// | Operation     | Semantics/RTL             | Assembly          |
  /// |---------------|---------------------------|-------------------|
  /// | Store Indexed | `Mem[r[b] + r[x]] ← r[d]` | `st rd, (rb, rx)` |
  StoreIndexed = 0x4,

  /// A placeholder opcode with no specific functionality defined.
  ///
  /// | Operation           | Semantics/RTL             | Assembly     |
  /// |---------------------|---------------------------|--------------|
  /// | Move                | `r[d] ← r[s]`             | `mov rs, rd` |
  /// | Add                 | `r[d] ← r[d] + r[s]`      | `add rs, rd` |
  /// | Logical AND         | `r[d] ← r[d] & r[s]`      | `and rs, rd` |
  /// | Increment           | `r[d] ← r[d] + 1`         | `inc rd`     |
  /// | Increment Address   | `r[d] ← r[d] + 4`         | `inca rd`    |
  /// | Decrement           | `r[d] ← r[d] − 1`         | `dec rd`     |
  /// | Decrement Address   | `r[d] ← r[d] − 4`         | `deca rd`    |
  /// | Logical NOT         | `r[d] ← ~r[d]`            | `not rd`     |
  /// | Get Program Counter | `r[d] ← pc + (o = 2 × p)` | `gpc $o, rd` |
  Miscellaneous = 0x6,

  /// | Operation          | Semantics/RTL                | Assembly              |
  /// |--------------------|------------------------------|-----------------------|
  /// | Shift Left Logical | `r[d] ← r[d] << vv`          | `shl $vv, rd`         |
  /// | Shift Right Logical| `r[d] ← r[d] >> -vv` (if vv < 0)| `shr $-vv, rd`      |
  ///
  /// - If `vv` is positive, performs a left logical shift (`shl`).
  /// - If `vv` is negative, performs a right logical shift (`shr`) using the absolute value of `vv`.
  Shift = 0x7,

  /// Performs a conditional branch to the target address based on some condition.
  ///
  /// | Operation | Semantics/RTL          | Assembly       |
  /// |-----------|------------------------|----------------|
  /// | Branch    | If `cond`, PC ← target | `bcond target` |
  Branch = 0x8,

  /// Branches if the values are equal.
  ///
  /// | Operation     | Semantics/RTL                 | Assembly             |
  /// |---------------|-------------------------------|----------------------|
  /// | BranchIfEqual | If `r[s] = r[t]`, PC ← target | `beq rs, rt, target` |
  BranchIfEqual = 0x9,

  /// Branches if one value is greater than another.
  ///
  /// | Operation       | Semantics/RTL                 | Assembly             |
  /// |-----------------|-------------------------------|----------------------|
  /// | BranchIfGreater | If `r[s] > r[t]`, PC ← target | `bgt rs, rt, target` |
  BranchIfGreater = 0xA,

  /// Jumps to a specified immediate address.
  ///
  /// | Operation      | Semantics/RTL | Assembly   |
  /// |----------------|---------------|------------|
  /// | Jump Immediate | PC ← target   | `j target` |
  JumpImmediate = 0xB,

  /// Jumps to an address calculated using a base address and offset.
  ///
  /// | Operation        | Semantics/RTL      | Assembly       |
  /// |------------------|--------------------|----------------|
  /// | Jump Base+Offset | PC ← r\[b\] + offset | `j offset(rb)` |
  JumpBaseOff = 0xC,

  /// Jumps to an address stored in a base register.
  ///
  /// | Operation     | Semantics/RTL | Assembly |
  /// |---------------|---------------|----------|
  /// | Jump Indirect | PC ← r\[b\]     | `j (rb)` |
  JumpIndirBaseOff = 0xD,

  /// Jumps to an address stored in a base register and indexed by another register.
  ///
  /// | Operation           | Semantics/RTL    | Assembly     |
  /// |---------------------|------------------|--------------|
  /// | Jump Indirect Index | PC ← r\[b\] + r\[x\] | `j (rb, rx)` |
  JumpIndirIndex = 0xE,

  /// Can either do nothing, or halt the program.
  ///
  /// | Operation | Semantics/RTL    | Assembly |
  /// |-----------|------------------|----------|
  /// | halt      | (stop execution) | halt     |
  /// | nop       | (do nothing)     | nop      |
  Nop = 0xF,
}

impl From<u8> for Opcode {
  fn from(maybe_nibble: u8) -> Self {
    match maybe_nibble & 0x0F {
      0x0 => Self::LoadImmediate,
      0x1 => Self::LoadBaseOff,
      0x2 => Self::LoadIndexed,
      0x3 => Self::StoreBaseOff,
      0x4 => Self::StoreIndexed,
      0x5 => panic!("opcode `0x5` does not exist"),
      0x6 => Self::Miscellaneous,
      0x7 => Self::Shift,
      0x8 => Self::Branch,
      0x9 => Self::BranchIfEqual,
      0xA => Self::BranchIfGreater,
      0xB => Self::JumpImmediate,
      0xC => Self::JumpBaseOff,
      0xD => Self::JumpIndirBaseOff,
      0xE => Self::JumpIndirIndex,
      0xF => Self::Nop,
      _ => unreachable!(),
    }
  }
}