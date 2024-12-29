/// A region of instructions
pub trait Region {
  fn instructions(&self) -> &[u8];
}

/// A `Chunk` is a single region of instructions that our virtual machine may
/// execute
pub struct Chunk {
  instructions: Vec<u8>,
}

impl From<Vec<u8>> for Chunk {
  fn from(instructions: Vec<u8>) -> Self {
    Self { instructions }
  }
}

impl Region for Chunk {
  fn instructions(&self) -> &[u8] {
    &self.instructions
  }
}
