use std::fmt;

use serde::Serialize;

use crate::error::{Error, Result};

use super::MAGIC_MTSP;

#[derive(Debug, Clone, Copy, Serialize)]
pub struct MTSPHeader {
    pub version: u32,
    pub reserved_1: u32,
    pub reserved_2: u32,
}

impl MTSPHeader {
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < 16 {
            return Err(Error::InvalidTrace(
                "capture data too small for MTSP header",
            ));
        }
        if data.get(..4) != Some(MAGIC_MTSP.as_slice()) {
            return Err(Error::InvalidTrace("capture data did not start with MTSP"));
        }
        Ok(Self {
            version: read_u32(data, 4)?,
            reserved_1: read_u32(data, 8)?,
            reserved_2: read_u32(data, 12)?,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum RecordType {
    C,
    C3ul,
    CS,
    CSuwuw,
    Ct,
    Ctt,
    CtU,
    Ctulul,
    CU,
    Cui,
    Cul,
    Culul,
    Cut,
    Cuw,
    CUUU,
    /// `CUt`: an object's reference to a bundle blob by key (address, then the
    /// key as a C string). Specialized `MTLFunction`s carry their descriptor
    /// this way instead of a `CS` name record.
    CUt,
    Ci,
    CiulSl,
    Ciulul,
    Unknown,
}

impl fmt::Display for RecordType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = match self {
            Self::C => "C",
            Self::C3ul => "C@3ul@3ul",
            Self::CS => "CS",
            Self::CSuwuw => "CSuwuw",
            Self::Ct => "Ct",
            Self::Ctt => "Ctt",
            Self::CtU => "CtU<b>ulul",
            Self::Ctulul => "Ctulul",
            Self::CU => "CU",
            Self::Cui => "Cui",
            Self::Cul => "Cul",
            Self::Culul => "Culul",
            Self::Cut => "Cut",
            Self::Cuw => "Cuw",
            Self::CUUU => "CUUU",
            Self::CUt => "CUt",
            Self::Ci => "Ci",
            Self::CiulSl => "CiulSl",
            Self::Ciulul => "Ciulul",
            Self::Unknown => "Unknown",
        };
        f.write_str(value)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct MTSPRecord {
    pub record_type: RecordType,
    pub offset: usize,
    pub size: usize,
    pub label: Option<String>,
    pub address: Option<u64>,
    pub function_address: Option<u64>,
    #[serde(skip_serializing)]
    pub data: Vec<u8>,
}

impl MTSPRecord {
    pub fn parse_stream(data: &[u8]) -> Result<Vec<Self>> {
        let mut offset = 0usize;
        if data.get(..4) == Some(MAGIC_MTSP.as_slice()) {
            MTSPHeader::parse(data)?;
            offset = 16;
        }

        let mut records = Vec::new();
        while offset + 8 <= data.len() {
            let record_size =
                u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
            if record_size == 0 || record_size > 0x400000 || offset + record_size > data.len() {
                offset += 4;
                continue;
            }

            let record_data = data[offset..offset + record_size].to_vec();
            let record_type = detect_record_type(&record_data);
            let mut record = MTSPRecord {
                record_type,
                offset,
                size: record_size,
                label: None,
                address: None,
                function_address: None,
                data: record_data,
            };

            match record.record_type {
                RecordType::CS => record.parse_cs_record(),
                RecordType::CUt => record.parse_cut_blob_record(),
                RecordType::CSuwuw => record.parse_csuwuw_record(),
                RecordType::CiulSl => record.parse_ciulsl_record(),
                RecordType::CU | RecordType::Cut => record.parse_cu_record(),
                RecordType::Cui => record.populate_cui_record(),
                RecordType::Ciulul => record.populate_ciulul_record(),
                RecordType::Culul => record.parse_culul_record(),
                _ => {}
            }

            records.push(record);
            offset += record_size;
        }

        Ok(records)
    }

    /// Parse the nested sub-records embedded inside a bulk record's payload.
    ///
    /// Bulk records (large CtU/Unknown records that hold collections of labeled
    /// resources, dispatches, etc.) carry a stream of sub-records *inside* their
    /// payload. Each sub-record uses the same framing as a top-level MTSP record
    /// (`u32 size` + tag detected in the first 128 bytes), but with an extended
    /// header inserted between the size and the tag:
    ///
    /// ```text
    /// +0   u32  sub-record size (includes itself)
    /// +4   [4]  magic: `?? YY 0xff 0xff` where YY is 0xc0 or 0xd8
    /// +8   [24] zero-padding
    /// +32  u32  count / inline-payload-prefix
    /// +36  [4]  type tag (CS\0\0, Cuw\0, Ci\0\0, CUUU, ...)
    /// +40  ...  type-specific payload
    /// ```
    ///
    /// `parse_stream` can't anchor onto these sub-records reliably because the
    /// bulk record's prelude length isn't fixed across record types and the
    /// 4-byte stride doesn't line up with the sub-records' absolute alignment.
    /// We instead scan for the magic header byte-by-byte, validate the framing
    /// (sane size + 24-zero pad), and parse the sub-record by handing its bytes
    /// back to the standard parser.
    pub fn parse_subrecords(data: &[u8]) -> Vec<Self> {
        let mut records = Vec::new();
        let mut i = 4;
        while i + 32 <= data.len() {
            let yy = data[i + 1];
            if (yy != 0xc0 && yy != 0xd8) || data[i + 2] != 0xff || data[i + 3] != 0xff {
                i += 1;
                continue;
            }
            let size = u32::from_le_bytes(data[i - 4..i].try_into().unwrap()) as usize;
            let start = i - 4;
            if size < 0x28
                || size > 0x10000
                || start + size > data.len()
                || !data[i + 4..i + 4 + 24].iter().all(|&byte| byte == 0)
            {
                i += 1;
                continue;
            }

            let sub = data[start..start + size].to_vec();
            let record_type = detect_record_type(&sub);
            let mut record = MTSPRecord {
                record_type,
                offset: start,
                size,
                label: None,
                address: None,
                function_address: None,
                data: sub,
            };
            match record.record_type {
                RecordType::CS => record.parse_cs_record(),
                RecordType::CUt => record.parse_cut_blob_record(),
                RecordType::CSuwuw => record.parse_csuwuw_record(),
                RecordType::CiulSl => record.parse_ciulsl_record(),
                RecordType::CU | RecordType::Cut => record.parse_cu_record(),
                RecordType::Cui => record.populate_cui_record(),
                RecordType::Ciulul => record.populate_ciulul_record(),
                RecordType::Culul => record.parse_culul_record(),
                _ => {}
            }
            records.push(record);
            i = start + size + 4;
        }
        records
    }

    pub fn parse_ct_record(&self) -> Result<CtRecord> {
        if self.record_type != RecordType::Ct {
            return Err(Error::InvalidTrace("record was not a Ct record"));
        }
        let Some(base) = find_bytes(&self.data, b"Ct\0\0") else {
            return Err(Error::InvalidTrace("Ct marker not found"));
        };
        if base + 28 > self.data.len() {
            return Err(Error::InvalidTrace("Ct record too small"));
        }

        let binding_count = read_u32(&self.data, base + 20)?;
        let stride = read_u32(&self.data, base + 24)?;

        let mut buffer_bindings = Vec::new();
        let mut resource_bindings = Vec::new();
        if binding_count > 0 && stride == 8 {
            let bindings_offset = base + 28;
            let bytes_needed = binding_count as usize * 8;
            if bindings_offset + bytes_needed <= self.data.len() {
                for index in 0..binding_count as usize {
                    let addr = read_u64(&self.data, bindings_offset + index * 8)?;
                    buffer_bindings.push(addr);
                    resource_bindings.push(ResourceBinding {
                        address: addr,
                        index,
                        usage: MTLResourceUsage::READ | MTLResourceUsage::WRITE,
                    });
                }
            }
        }

        Ok(CtRecord {
            record_size: self.size as u32,
            command_flags: read_u32(&self.data, 0).unwrap_or_default(),
            pipeline_addr: read_u64(&self.data, base + 4)?,
            function_addr: read_u64(&self.data, base + 12)?,
            binding_count,
            stride,
            buffer_bindings,
            resource_bindings,
        })
    }

    pub fn parse_ctt_record(&self) -> Result<CttRecord> {
        if self.record_type != RecordType::Ctt {
            return Err(Error::InvalidTrace("record was not a Ctt record"));
        }
        let Some(base) = find_bytes(&self.data, b"Ctt\0") else {
            return Err(Error::InvalidTrace("Ctt marker not found"));
        };
        // Fixed fields end with the pipeline address at base + 0x20; the
        // binding table header (base + 0x28) is absent when there are no
        // bindings (`newComputePipelineStateWithFunction:` records are 76
        // bytes).
        if base + 0x28 > self.data.len() {
            return Err(Error::InvalidTrace("Ctt record too small"));
        }

        let binding_count = read_u32(&self.data, base + 0x28).unwrap_or(0);
        let stride = read_u32(&self.data, base + 0x2c).unwrap_or(0);
        let mut buffer_bindings = Vec::new();
        let mut resource_bindings = Vec::new();
        if binding_count > 0 && stride == 8 {
            let bindings_offset = base + 0x30;
            let bytes_needed = binding_count as usize * 8;
            if bindings_offset + bytes_needed <= self.data.len() {
                for index in 0..binding_count as usize {
                    let addr = read_u64(&self.data, bindings_offset + index * 8)?;
                    buffer_bindings.push(addr);
                    resource_bindings.push(ResourceBinding {
                        address: addr,
                        index,
                        usage: MTLResourceUsage::READ | MTLResourceUsage::WRITE,
                    });
                }
            }
        }

        Ok(CttRecord {
            record_size: self.size as u32,
            command_flags: read_u32(&self.data, 4).unwrap_or_default(),
            device_addr: read_u64(&self.data, base + 4)?,
            function_addr: read_u64(&self.data, base + 12)?,
            pipeline_addr: read_u64(&self.data, base + 0x20)?,
            binding_count,
            stride,
            buffer_bindings,
            resource_bindings,
        })
    }

    pub fn parse_ci_record(&self) -> Result<CiRecord> {
        if self.record_type != RecordType::Ci {
            return Err(Error::InvalidTrace("record was not a Ci record"));
        }
        if self.data.len() < 52 {
            return Err(Error::InvalidTrace("Ci record too small"));
        }
        let record = CiRecord {
            record_size: read_u32(&self.data, 0)?,
            command_flags: read_u32(&self.data, 4)?,
            field_1: read_u32(&self.data, 0x20)?,
            icb_addr: read_u64(&self.data, 0x28)?,
            count: read_u32(&self.data, 0x30)?,
            field_2: read_u32(&self.data, 0x34)?,
        };
        if record.record_size != 52 {
            return Err(Error::InvalidTrace("unexpected Ci record size"));
        }
        Ok(record)
    }

    pub fn parse_culul_structured(&self) -> Result<CululRecord> {
        if self.record_type != RecordType::Culul {
            return Err(Error::InvalidTrace("record was not a Culul record"));
        }
        if self.data.len() < 0x58 {
            return Err(Error::InvalidTrace("Culul record too small"));
        }
        let array_count = read_u32(&self.data, 0x50)?;
        let mut array_addresses = Vec::new();
        for index in 0..array_count as usize {
            let offset = 0x58 + index * 8;
            if offset + 8 > self.data.len() {
                return Err(Error::InvalidTrace("Culul array ran out of bounds"));
            }
            array_addresses.push(read_u64(&self.data, offset)?);
        }
        Ok(CululRecord {
            record_size: read_u32(&self.data, 0)?,
            command_flags: read_u32(&self.data, 4)?,
            marker_count: read_u32(&self.data, 0x20)?,
            icb_addr: read_u64(&self.data, 0x28)?,
            field_1: read_u32(&self.data, 0x30)?,
            field_2: read_u32(&self.data, 0x34)?,
            field_3: read_u32(&self.data, 0x38)?,
            payload_size: read_u32(&self.data, 0x40)?,
            payload_addr: read_u64(&self.data, 0x48)?,
            array_count,
            array_stride: read_u32(&self.data, 0x54)?,
            array_addresses,
        })
    }

    pub fn parse_cul_record(&self) -> Result<CulRecord> {
        if self.record_type != RecordType::Cul {
            return Err(Error::InvalidTrace("record was not a Cul record"));
        }
        if self.data.len() < 0x34 {
            return Err(Error::InvalidTrace("Cul record too small"));
        }
        Ok(CulRecord {
            record_size: read_u32(&self.data, 0)?,
            command_flags: read_u32(&self.data, 4)?,
            marker_count: read_u32(&self.data, 0x20)?,
            buffer_addr: read_u64(&self.data, 0x28)?,
            field_1: read_u32(&self.data, 0x30)?,
        })
    }

    pub fn parse_cuw_record(&self) -> Result<CuwRecord> {
        if self.record_type != RecordType::Cuw {
            return Err(Error::InvalidTrace("record was not a Cuw record"));
        }
        if self.data.len() < 0x30 {
            return Err(Error::InvalidTrace("Cuw record too small"));
        }
        let is_extended = self
            .data
            .get(0x24..)
            .is_some_and(|tail| tail.starts_with(b"Cuwuw"));
        let buffer_addr = if is_extended {
            read_u64(&self.data, 0x2c)?
        } else {
            read_u64(&self.data, 0x28)?
        };
        let field_1 = if is_extended {
            0
        } else {
            read_u64(&self.data, 0x30).unwrap_or_default()
        };
        let field_2 = if is_extended {
            read_u32(&self.data, 0x34).unwrap_or_default()
        } else {
            0
        };
        Ok(CuwRecord {
            record_size: read_u32(&self.data, 0)?,
            command_flags: read_u32(&self.data, 4)?,
            marker_count: read_u32(&self.data, 0x20)?,
            buffer_addr,
            field_1,
            field_2,
        })
    }

    pub fn parse_cu_structured(&self) -> Result<CuRecord> {
        if self.record_type != RecordType::CU && self.record_type != RecordType::Cut {
            return Err(Error::InvalidTrace("record was not a CU/Cut record"));
        }
        let Some(base) =
            find_bytes(&self.data, b"CU\0\0").or_else(|| find_bytes(&self.data, b"Cut"))
        else {
            return Err(Error::InvalidTrace("CU/Cut marker not found"));
        };
        if base + 0x28 > self.data.len() {
            return Err(Error::InvalidTrace("CU/Cut record too small"));
        }
        Ok(CuRecord {
            record_size: read_u32(&self.data, 0)?,
            command_flags: read_u32(&self.data, 4)?,
            device_addr: read_u64(&self.data, base + 4)?,
            identifier: self.label.clone(),
            heap_addr: read_u64(&self.data, base + 0x20)?,
        })
    }

    pub fn parse_cui_record(&self) -> Result<CuiRecord> {
        if self.record_type != RecordType::Cui {
            return Err(Error::InvalidTrace("record was not a Cui record"));
        }
        let Some(base) = find_bytes(&self.data, b"Cui\0") else {
            return Err(Error::InvalidTrace("Cui marker not found"));
        };
        Ok(CuiRecord {
            record_size: read_u32(&self.data, 0)?,
            command_flags: read_u32(&self.data, 4).unwrap_or_default(),
            shared_event_addr: read_u64(&self.data, base + 4)?,
        })
    }

    pub fn parse_ciulul_record(&self) -> Result<CiululRecord> {
        if self.record_type != RecordType::Ciulul {
            return Err(Error::InvalidTrace("record was not a Ciulul record"));
        }
        let Some(base) = find_bytes(&self.data, b"Ciulul") else {
            return Err(Error::InvalidTrace("Ciulul marker not found"));
        };
        Ok(CiululRecord {
            record_size: self.size as u32,
            command_flags: read_u32(&self.data, 4).unwrap_or_default(),
            icb_addr: read_u64(&self.data, base + 8).ok(),
            count: read_u32(&self.data, base + 16).ok(),
        })
    }

    pub fn parse_ctu_record(&self) -> Result<CtURecord> {
        if self.record_type != RecordType::CtU {
            return Err(Error::InvalidTrace("record was not a CtU record"));
        }
        let Some(base) = find_bytes(&self.data, b"CtU<b>ulul") else {
            return Err(Error::InvalidTrace("CtU marker not found"));
        };
        let address = read_u64(&self.data, base + 20)?;
        let name =
            read_c_string(&self.data, base + 28).ok_or(Error::InvalidTrace("CtU name missing"))?;
        Ok(CtURecord {
            record_size: self.size as u32,
            address,
            name,
        })
    }

    pub fn parse_ctulul_record(&self) -> Result<CttRecord> {
        if self.record_type != RecordType::Ctulul {
            return Err(Error::InvalidTrace("record was not a Ctulul record"));
        }
        let Some(base) = find_bytes(&self.data, b"Ctulul") else {
            return Err(Error::InvalidTrace("Ctulul marker not found"));
        };
        if base + 52 > self.data.len() {
            return Err(Error::InvalidTrace("Ctulul record too small"));
        }
        let binding_count = read_u32(&self.data, base + 44)?;
        let buffer_start = base + 52;
        let mut buffer_bindings = Vec::new();
        for index in 0..binding_count as usize {
            let offset = buffer_start + index * 8;
            if offset + 8 > self.data.len() {
                break;
            }
            buffer_bindings.push(read_u64(&self.data, offset)?);
        }
        let resource_bindings = buffer_bindings
            .iter()
            .enumerate()
            .map(|(index, address)| ResourceBinding {
                address: *address,
                index,
                usage: MTLResourceUsage::READ | MTLResourceUsage::WRITE,
            })
            .collect();
        Ok(CttRecord {
            record_size: self.size as u32,
            command_flags: read_u32(&self.data, 4).unwrap_or_default(),
            device_addr: 0,
            function_addr: 0,
            pipeline_addr: read_u64(&self.data, base + 8)?,
            binding_count,
            stride: 8,
            buffer_bindings,
            resource_bindings,
        })
    }

    pub fn parse_c_record(&self) -> Result<CRecord> {
        if self.record_type != RecordType::C {
            return Err(Error::InvalidTrace("record was not a C record"));
        }
        let Some(base) = find_bytes(&self.data, b"C\0\0\0") else {
            return Err(Error::InvalidTrace("C marker not found"));
        };
        Ok(CRecord {
            record_size: self.size as u32,
            command_flags: read_u32(&self.data, 4).unwrap_or_default(),
            encoder_addr: read_u64(&self.data, base + 8)?,
        })
    }

    pub fn parse_dispatch_record(&self) -> Result<CDispatchRecord> {
        if self.record_type != RecordType::C3ul {
            return Err(Error::InvalidTrace("record was not a dispatch record"));
        }
        if self.data.len() < 0x68 {
            return Err(Error::InvalidTrace("dispatch record too small"));
        }
        Ok(CDispatchRecord {
            record_size: self.size as u32,
            command_flags: read_u32(&self.data, 4).unwrap_or_default(),
            encoder_id: read_u64(&self.data, 0x30)?,
            grid_size: [
                read_u64(&self.data, 0x38)? as u32,
                read_u64(&self.data, 0x40)? as u32,
                read_u64(&self.data, 0x48)? as u32,
            ],
            group_size: [
                read_u64(&self.data, 0x50)? as u32,
                read_u64(&self.data, 0x58)? as u32,
                read_u64(&self.data, 0x60)? as u32,
            ],
        })
    }

    fn parse_csuwuw_record(&mut self) {
        let Some(base) = find_bytes(&self.data, b"CSuwuw") else {
            return;
        };
        let addr_start = base + 9;
        if addr_start + 8 > self.data.len() {
            return;
        }
        self.address = read_u64(&self.data, addr_start).ok();
        let mut str_start = addr_start + 8;
        while str_start < self.data.len() && self.data[str_start] == 0 {
            str_start += 1;
        }
        self.label = read_c_string(&self.data, str_start);
    }

    /// For a factory call (`newFunctionWithName:`, `newFunctionWithDescriptor:`
    /// and the like), the object it returned: the address after the `t`
    /// marker that follows the arguments. `setLabel:`-style records have none.
    pub fn returned_address(&self) -> Option<u64> {
        let label_end = self
            .label
            .as_ref()
            .and_then(|label| find_bytes(&self.data, label.as_bytes()).map(|start| start + label.len()))?;
        let marker = find_bytes(&self.data[label_end..], b"t\0\0\0")? + label_end;
        if self.data[label_end..marker].iter().any(|&byte| byte != 0) {
            return None;
        }
        read_u64(&self.data, marker + 4).ok().filter(|&address| address != 0)
    }

    fn parse_cut_blob_record(&mut self) {
        let Some(base) = find_bytes(&self.data, b"CUt\0") else {
            return;
        };
        let addr_start = base + 4;
        if addr_start + 8 > self.data.len() {
            return;
        }
        self.address = read_u64(&self.data, addr_start).ok();
        self.label = read_c_string(&self.data, addr_start + 8);
    }

    fn parse_cs_record(&mut self) {
        let Some(base) = find_bytes(&self.data, b"CS\0\0") else {
            return;
        };
        let addr_start = base + 4;
        if addr_start + 8 > self.data.len() {
            return;
        }
        self.address = read_u64(&self.data, addr_start).ok();
        self.label = read_c_string(&self.data, addr_start + 8);
    }

    fn parse_ciulsl_record(&mut self) {
        let Some(base) = find_bytes(&self.data, b"CiulSl") else {
            return;
        };
        self.function_address = read_u64(&self.data, base + 8).ok();
    }

    fn parse_cu_record(&mut self) {
        for start in 0..self.data.len().saturating_sub(8) {
            let slice = &self.data[start..self.data.len().min(start + 64)];
            if looks_like_hex_label(slice) {
                let end = slice
                    .iter()
                    .position(|byte| !is_hex_or_dash(*byte))
                    .unwrap_or(slice.len());
                if end >= 8 {
                    self.label = Some(String::from_utf8_lossy(&slice[..end]).into_owned());
                    return;
                }
            }
        }
    }

    fn populate_cui_record(&mut self) {
        let Some(base) = find_bytes(&self.data, b"Cui\0") else {
            return;
        };
        self.address = read_u64(&self.data, base + 4).ok();
    }

    fn populate_ciulul_record(&mut self) {
        let Some(base) = find_bytes(&self.data, b"Ciulul") else {
            return;
        };
        self.address = read_u64(&self.data, base + 8).ok();
    }

    fn parse_culul_record(&mut self) {
        let Some(base) = find_bytes(&self.data, b"Culul") else {
            return;
        };
        self.address = read_u64(&self.data, base + 5).ok();
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct CtRecord {
    pub record_size: u32,
    pub command_flags: u32,
    pub pipeline_addr: u64,
    pub function_addr: u64,
    pub binding_count: u32,
    pub stride: u32,
    pub buffer_bindings: Vec<u64>,
    pub resource_bindings: Vec<ResourceBinding>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CiRecord {
    pub record_size: u32,
    pub command_flags: u32,
    pub field_1: u32,
    pub icb_addr: u64,
    pub count: u32,
    pub field_2: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct CululRecord {
    pub record_size: u32,
    pub command_flags: u32,
    pub marker_count: u32,
    pub icb_addr: u64,
    pub field_1: u32,
    pub field_2: u32,
    pub field_3: u32,
    pub payload_size: u32,
    pub payload_addr: u64,
    pub array_count: u32,
    pub array_stride: u32,
    pub array_addresses: Vec<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CulRecord {
    pub record_size: u32,
    pub command_flags: u32,
    pub marker_count: u32,
    pub buffer_addr: u64,
    pub field_1: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct CuwRecord {
    pub record_size: u32,
    pub command_flags: u32,
    pub marker_count: u32,
    pub buffer_addr: u64,
    pub field_1: u64,
    pub field_2: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct CuRecord {
    pub record_size: u32,
    pub command_flags: u32,
    pub device_addr: u64,
    pub identifier: Option<String>,
    pub heap_addr: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CuiRecord {
    pub record_size: u32,
    pub command_flags: u32,
    pub shared_event_addr: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CiululRecord {
    pub record_size: u32,
    pub command_flags: u32,
    pub icb_addr: Option<u64>,
    pub count: Option<u32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CtURecord {
    pub record_size: u32,
    pub address: u64,
    pub name: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct CRecord {
    pub record_size: u32,
    pub command_flags: u32,
    pub encoder_addr: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CDispatchRecord {
    pub record_size: u32,
    pub command_flags: u32,
    pub encoder_id: u64,
    pub grid_size: [u32; 3],
    pub group_size: [u32; 3],
}

#[derive(Debug, Clone, Serialize)]
pub struct CttRecord {
    pub record_size: u32,
    pub command_flags: u32,
    pub device_addr: u64,
    pub function_addr: u64,
    pub pipeline_addr: u64,
    pub binding_count: u32,
    pub stride: u32,
    pub buffer_bindings: Vec<u64>,
    pub resource_bindings: Vec<ResourceBinding>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct MTLResourceUsage(u8);

impl MTLResourceUsage {
    pub const READ: Self = Self(0x01);
    pub const WRITE: Self = Self(0x02);
    pub const SAMPLE: Self = Self(0x04);

    pub fn bits(self) -> u8 {
        self.0
    }

    pub fn contains(self, rhs: Self) -> bool {
        self.0 & rhs.0 == rhs.0
    }
}

impl std::ops::BitOr for MTLResourceUsage {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl fmt::Display for MTLResourceUsage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts = Vec::new();
        if self.contains(Self::READ) {
            parts.push("read");
        }
        if self.contains(Self::WRITE) {
            parts.push("write");
        }
        if self.contains(Self::SAMPLE) {
            parts.push("sample");
        }
        if parts.is_empty() {
            parts.push("none");
        }
        f.write_str(&parts.join("|"))
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ResourceBinding {
    pub address: u64,
    pub index: usize,
    pub usage: MTLResourceUsage,
}

fn detect_record_type(data: &[u8]) -> RecordType {
    if data.len() < 16 {
        return RecordType::Unknown;
    }

    for i in 4..data.len().min(128) {
        if starts_with_at(data, i, b"Culul") {
            return RecordType::Culul;
        }
        if starts_with_at(data, i, b"Cuw") {
            return RecordType::Cuw;
        }
        if starts_with_at(data, i, b"Cut") {
            return RecordType::Cut;
        }
        if starts_with_at(data, i, b"Ciulul") {
            return RecordType::Ciulul;
        }
        if starts_with_at(data, i, b"CtU<b>ulul") {
            return RecordType::CtU;
        }
        if starts_with_at(data, i, b"Ctulul") {
            return RecordType::Ctulul;
        }
        if starts_with_at(data, i, b"C@3ul@3ul\0") {
            return RecordType::C3ul;
        }
        if starts_with_at(data, i, b"C\0\0\0") {
            return RecordType::C;
        }
        if starts_with_at(data, i, b"CSuwuw") {
            return RecordType::CSuwuw;
        }
        if starts_with_at(data, i, b"CS") && data.get(i + 2) == Some(&0) {
            return RecordType::CS;
        }
        if starts_with_at(data, i, b"Ctt") {
            return RecordType::Ctt;
        }
        if starts_with_at(data, i, b"Ct")
            && data
                .get(i + 2)
                .is_some_and(|next| *next != b't' && *next != b'u')
            && data.get(i + 2) == Some(&0)
        {
            return RecordType::Ct;
        }
        if starts_with_at(data, i, b"CiulSl") {
            return RecordType::CiulSl;
        }
        if starts_with_at(data, i, b"Cul") {
            return RecordType::Cul;
        }
        if starts_with_at(data, i, b"CUUU") {
            return RecordType::CUUU;
        }
        if starts_with_at(data, i, b"CUt\0") {
            return RecordType::CUt;
        }
        if starts_with_at(data, i, b"CU") && data.get(i + 2) == Some(&0) {
            return RecordType::CU;
        }
        if starts_with_at(data, i, b"Ci") && data.get(i + 2) == Some(&0) {
            if starts_with_at(data, i, b"Cui") {
                return RecordType::Cui;
            }
            return RecordType::Ci;
        }
    }

    RecordType::Unknown
}

fn looks_like_hex_label(data: &[u8]) -> bool {
    let mut count = 0usize;
    for byte in data.iter().take(32) {
        if is_hex(*byte) {
            count += 1;
        } else if *byte == 0 {
            break;
        } else {
            return false;
        }
    }
    count >= 8
}

fn is_hex_or_dash(byte: u8) -> bool {
    is_hex(byte) || byte == b'-'
}

fn is_hex(byte: u8) -> bool {
    byte.is_ascii_hexdigit()
}

fn starts_with_at(data: &[u8], offset: usize, needle: &[u8]) -> bool {
    data.get(offset..offset + needle.len()) == Some(needle)
}

fn find_bytes(data: &[u8], needle: &[u8]) -> Option<usize> {
    data.windows(needle.len())
        .position(|window| window == needle)
}

fn read_c_string(data: &[u8], offset: usize) -> Option<String> {
    let tail = data.get(offset..)?;
    let end = tail.iter().position(|byte| *byte == 0)?;
    if end == 0 {
        return None;
    }
    Some(String::from_utf8_lossy(&tail[..end]).into_owned())
}

fn read_u32(data: &[u8], offset: usize) -> Result<u32> {
    let bytes = data
        .get(offset..offset + 4)
        .ok_or(Error::InvalidTrace("short u32 read"))?;
    Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
}

fn read_u64(data: &[u8], offset: usize) -> Result<u64> {
    let bytes = data
        .get(offset..offset + 8)
        .ok_or(Error::InvalidTrace("short u64 read"))?;
    Ok(u64::from_le_bytes(bytes.try_into().unwrap()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_mtsp_header() {
        let data = [b'M', b'T', b'S', b'P', 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0];
        let header = MTSPHeader::parse(&data).unwrap();
        assert_eq!(header.version, 1);
        assert_eq!(header.reserved_1, 2);
        assert_eq!(header.reserved_2, 3);
    }

    #[test]
    fn parses_ct_record() {
        let mut data = vec![0u8; 100];
        let marker_offset = 16;
        data[marker_offset..marker_offset + 4].copy_from_slice(b"Ct\0\0");

        let pipeline_addr = 0x1122334455667788u64;
        let function_addr = 0x8877665544332211u64;
        let binding_count = 2u32;
        let stride = 8u32;
        let binding_1 = 0xAABBCCDDEEFF0011u64;
        let binding_2 = 0x1100FFEEDDCCBBAAu64;

        data[marker_offset + 4..marker_offset + 12].copy_from_slice(&pipeline_addr.to_le_bytes());
        data[marker_offset + 12..marker_offset + 20].copy_from_slice(&function_addr.to_le_bytes());
        data[marker_offset + 20..marker_offset + 24].copy_from_slice(&binding_count.to_le_bytes());
        data[marker_offset + 24..marker_offset + 28].copy_from_slice(&stride.to_le_bytes());
        data[marker_offset + 28..marker_offset + 36].copy_from_slice(&binding_1.to_le_bytes());
        data[marker_offset + 36..marker_offset + 44].copy_from_slice(&binding_2.to_le_bytes());

        let record = MTSPRecord {
            record_type: RecordType::Ct,
            offset: 0,
            size: data.len(),
            label: None,
            address: None,
            function_address: None,
            data,
        };

        let ct = record.parse_ct_record().unwrap();
        assert_eq!(ct.pipeline_addr, pipeline_addr);
        assert_eq!(ct.function_addr, function_addr);
        assert_eq!(ct.binding_count, binding_count);
        assert_eq!(ct.buffer_bindings, vec![binding_1, binding_2]);
    }

    #[test]
    fn parses_csuwuw_record() {
        let mut data = vec![0u8; 100];
        let marker_offset = 10;
        data[marker_offset..marker_offset + 6].copy_from_slice(b"CSuwuw");

        let func_addr = 0x00CAFEBABE112233u64;
        let addr_offset = marker_offset + 9;
        data[addr_offset..addr_offset + 8].copy_from_slice(&func_addr.to_le_bytes());
        let string_start = addr_offset + 10;
        data[string_start..string_start + 12].copy_from_slice(b"MyKernelFunc");
        data[string_start + 12] = 0;

        let mut record = MTSPRecord {
            record_type: RecordType::CSuwuw,
            offset: 0,
            size: data.len(),
            label: None,
            address: None,
            function_address: None,
            data,
        };
        record.parse_csuwuw_record();
        assert_eq!(record.address, Some(func_addr));
        assert_eq!(record.label.as_deref(), Some("MyKernelFunc"));
    }

    #[test]
    fn parses_ciulsl_record() {
        let mut data = vec![0u8; 64];
        let marker_offset = 10;
        data[marker_offset..marker_offset + 6].copy_from_slice(b"CiulSl");
        let func_addr = 0xDEADBEEFu64;
        data[marker_offset + 8..marker_offset + 16].copy_from_slice(&func_addr.to_le_bytes());

        let mut record = MTSPRecord {
            record_type: RecordType::CiulSl,
            offset: 0,
            size: data.len(),
            label: None,
            address: None,
            function_address: None,
            data,
        };
        record.parse_ciulsl_record();
        assert_eq!(record.function_address, Some(func_addr));
    }

    #[test]
    fn parses_records_from_stream() {
        let mut stream = Vec::new();
        stream.extend_from_slice(b"MTSP");
        stream.extend_from_slice(&1u32.to_le_bytes());
        stream.extend_from_slice(&0u32.to_le_bytes());
        stream.extend_from_slice(&0u32.to_le_bytes());

        let mut record = vec![0u8; 32];
        record[0..4].copy_from_slice(&(32u32).to_le_bytes());
        record[8..12].copy_from_slice(b"CS\0\0");
        let addr = 0x1234u64;
        record[12..20].copy_from_slice(&addr.to_le_bytes());
        record[20..27].copy_from_slice(b"Kernel\0");
        stream.extend_from_slice(&record);

        let records = MTSPRecord::parse_stream(&stream).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].record_type, RecordType::CS);
        assert_eq!(records[0].address, Some(addr));
        assert_eq!(records[0].label.as_deref(), Some("Kernel"));
    }

    #[test]
    fn parses_ctt_record() {
        let mut data = vec![0u8; 96];
        let marker_offset = 16;
        data[marker_offset..marker_offset + 4].copy_from_slice(b"Ctt\0");
        let device_addr = 0x1111u64;
        let function_addr = 0x2222u64;
        let pipeline_addr = 0x3333u64;
        let binding = 0x4444u64;
        data[marker_offset + 4..marker_offset + 12].copy_from_slice(&device_addr.to_le_bytes());
        data[marker_offset + 12..marker_offset + 20].copy_from_slice(&function_addr.to_le_bytes());
        data[marker_offset + 0x20..marker_offset + 0x28]
            .copy_from_slice(&pipeline_addr.to_le_bytes());
        data[marker_offset + 0x28..marker_offset + 0x2c].copy_from_slice(&1u32.to_le_bytes());
        data[marker_offset + 0x2c..marker_offset + 0x30].copy_from_slice(&8u32.to_le_bytes());
        data[marker_offset + 0x30..marker_offset + 0x38].copy_from_slice(&binding.to_le_bytes());

        let record = MTSPRecord {
            record_type: RecordType::Ctt,
            offset: 0,
            size: data.len(),
            label: None,
            address: None,
            function_address: None,
            data,
        };
        let ctt = record.parse_ctt_record().unwrap();
        assert_eq!(ctt.device_addr, device_addr);
        assert_eq!(ctt.function_addr, function_addr);
        assert_eq!(ctt.pipeline_addr, pipeline_addr);
        assert_eq!(ctt.buffer_bindings, vec![binding]);
    }

    #[test]
    fn parses_ctu_record() {
        let mut data = vec![0u8; 80];
        let marker_offset = 10;
        data[marker_offset..marker_offset + 10].copy_from_slice(b"CtU<b>ulul");
        let address = 0xABCDEFu64;
        data[marker_offset + 20..marker_offset + 28].copy_from_slice(&address.to_le_bytes());
        data[marker_offset + 28..marker_offset + 38].copy_from_slice(b"MTLBuffer\0");

        let record = MTSPRecord {
            record_type: RecordType::CtU,
            offset: 0,
            size: data.len(),
            label: None,
            address: None,
            function_address: None,
            data,
        };
        let ctu = record.parse_ctu_record().unwrap();
        assert_eq!(ctu.address, address);
        assert_eq!(ctu.name, "MTLBuffer");
    }

    #[test]
    fn parses_cu_record() {
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(&(64u32).to_le_bytes());
        data[4..8].copy_from_slice(&(0x55u32).to_le_bytes());
        let marker_offset = 0x18;
        data[marker_offset..marker_offset + 4].copy_from_slice(b"CU\0\0");
        data[marker_offset + 4..marker_offset + 12]
            .copy_from_slice(&0x1122334455667788u64.to_le_bytes());
        data[marker_offset + 0x20..marker_offset + 0x28]
            .copy_from_slice(&0x8877665544332211u64.to_le_bytes());

        let record = MTSPRecord {
            record_type: RecordType::CU,
            offset: 0,
            size: data.len(),
            label: Some("01234567-89ab-cdef".into()),
            address: None,
            function_address: None,
            data,
        };

        let cu = record.parse_cu_structured().unwrap();
        assert_eq!(cu.record_size, 64);
        assert_eq!(cu.command_flags, 0x55);
        assert_eq!(cu.device_addr, 0x1122334455667788);
        assert_eq!(cu.heap_addr, 0x8877665544332211);
        assert_eq!(cu.identifier.as_deref(), Some("01234567-89ab-cdef"));
    }

    #[test]
    fn parses_cui_record() {
        let mut data = vec![0u8; 32];
        data[0..4].copy_from_slice(&(32u32).to_le_bytes());
        data[4..8].copy_from_slice(&(0x22u32).to_le_bytes());
        let marker_offset = 0x10;
        data[marker_offset..marker_offset + 4].copy_from_slice(b"Cui\0");
        data[marker_offset + 4..marker_offset + 12]
            .copy_from_slice(&0xDEADBEEF00112233u64.to_le_bytes());

        let record = MTSPRecord {
            record_type: RecordType::Cui,
            offset: 0,
            size: data.len(),
            label: None,
            address: None,
            function_address: None,
            data,
        };

        let cui = record.parse_cui_record().unwrap();
        assert_eq!(cui.shared_event_addr, 0xDEADBEEF00112233);
    }

    #[test]
    fn parses_ciulul_record() {
        let mut data = vec![0u8; 48];
        data[0..4].copy_from_slice(&(48u32).to_le_bytes());
        data[4..8].copy_from_slice(&(0x33u32).to_le_bytes());
        let marker_offset = 0x10;
        data[marker_offset..marker_offset + 6].copy_from_slice(b"Ciulul");
        data[marker_offset + 8..marker_offset + 16]
            .copy_from_slice(&0xCAFEBABE11223344u64.to_le_bytes());
        data[marker_offset + 16..marker_offset + 20].copy_from_slice(&(7u32).to_le_bytes());

        let record = MTSPRecord {
            record_type: RecordType::Ciulul,
            offset: 0,
            size: data.len(),
            label: None,
            address: None,
            function_address: None,
            data,
        };

        let ciulul = record.parse_ciulul_record().unwrap();
        assert_eq!(ciulul.command_flags, 0x33);
        assert_eq!(ciulul.icb_addr, Some(0xCAFEBABE11223344));
        assert_eq!(ciulul.count, Some(7));
    }

    #[test]
    fn formats_resource_usage() {
        let usage = MTLResourceUsage::READ | MTLResourceUsage::WRITE;
        assert!(usage.contains(MTLResourceUsage::READ));
        assert!(usage.contains(MTLResourceUsage::WRITE));
        assert_eq!(usage.bits(), 0x03);
        assert_eq!(usage.to_string(), "read|write");
    }
}
