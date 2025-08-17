use anyhow::Result;
use std::sync::{Arc, Mutex};

/// Manages frame buffers for the compute output
pub struct FrameBuffer {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
    pub generation: u64,
}

impl FrameBuffer {
    pub fn new(width: u32, height: u32) -> Self {
        let size = (width * height * 4) as usize;
        Self {
            width,
            height,
            data: vec![0u8; size],
            generation: 0,
        }
    }

    pub fn update(&mut self, data: &[u8]) -> Result<()> {
        if data.len() != self.data.len() {
            return Err(anyhow::anyhow!(
                "Frame buffer size mismatch: expected {}, got {}",
                self.data.len(),
                data.len()
            ));
        }

        self.data.copy_from_slice(data);
        self.generation += 1;
        Ok(())
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        let size = (width * height * 4) as usize;
        self.data.resize(size, 0);
        self.generation += 1;
    }
}

/// Thread-safe frame buffer for sharing between compute and render threads
pub struct SharedFrameBuffer {
    buffer: Arc<Mutex<FrameBuffer>>,
}

impl SharedFrameBuffer {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            buffer: Arc::new(Mutex::new(FrameBuffer::new(width, height))),
        }
    }

    pub fn update(&self, data: &[u8]) -> Result<()> {
        let mut buffer = self.buffer.lock().unwrap();
        buffer.update(data)
    }

    pub fn get_data(&self) -> (Vec<u8>, u32, u32, u64) {
        let buffer = self.buffer.lock().unwrap();
        (
            buffer.data.clone(),
            buffer.width,
            buffer.height,
            buffer.generation,
        )
    }

    pub fn resize(&self, width: u32, height: u32) {
        let mut buffer = self.buffer.lock().unwrap();
        buffer.resize(width, height);
    }

    pub fn clone_handle(&self) -> Self {
        Self {
            buffer: Arc::clone(&self.buffer),
        }
    }
}
