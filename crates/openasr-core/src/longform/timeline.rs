#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimelineAnchor {
    pub processed_seconds: f32,
    pub original_seconds: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TimelineMap {
    anchors: Vec<TimelineAnchor>,
}

impl TimelineMap {
    pub fn identity() -> Self {
        Self {
            anchors: vec![
                TimelineAnchor {
                    processed_seconds: 0.0,
                    original_seconds: 0.0,
                },
                TimelineAnchor {
                    processed_seconds: 1.0,
                    original_seconds: 1.0,
                },
            ],
        }
    }

    pub fn from_anchors(mut anchors: Vec<TimelineAnchor>) -> Self {
        if anchors.is_empty() {
            return Self::identity();
        }
        anchors.sort_by(|left, right| {
            left.processed_seconds
                .partial_cmp(&right.processed_seconds)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        anchors.dedup_by(|left, right| left.processed_seconds == right.processed_seconds);
        if anchors.len() == 1 {
            anchors.push(TimelineAnchor {
                processed_seconds: anchors[0].processed_seconds + 1.0,
                original_seconds: anchors[0].original_seconds + 1.0,
            });
        }
        Self { anchors }
    }

    pub fn anchors(&self) -> &[TimelineAnchor] {
        &self.anchors
    }

    pub fn map_processed_to_original_seconds(&self, processed_seconds: f32) -> f32 {
        if !processed_seconds.is_finite() {
            return processed_seconds;
        }
        if processed_seconds <= self.anchors[0].processed_seconds {
            return extrapolate_with_pair(
                processed_seconds,
                self.anchors[0],
                self.anchors[1.min(self.anchors.len() - 1)],
            );
        }
        let last = self.anchors.len() - 1;
        if processed_seconds >= self.anchors[last].processed_seconds {
            return extrapolate_with_pair(
                processed_seconds,
                self.anchors[last.saturating_sub(1)],
                self.anchors[last],
            );
        }
        let mut low = 0usize;
        let mut high = last;
        while high - low > 1 {
            let mid = low + (high - low) / 2;
            if self.anchors[mid].processed_seconds <= processed_seconds {
                low = mid;
            } else {
                high = mid;
            }
        }
        let left = self.anchors[low];
        let right = self.anchors[high];
        let den = right.processed_seconds - left.processed_seconds;
        if den <= f32::EPSILON {
            return left.original_seconds;
        }
        let t = (processed_seconds - left.processed_seconds) / den;
        left.original_seconds + (right.original_seconds - left.original_seconds) * t
    }
}

fn extrapolate_with_pair(
    processed_seconds: f32,
    left: TimelineAnchor,
    right: TimelineAnchor,
) -> f32 {
    let den = right.processed_seconds - left.processed_seconds;
    if den.abs() <= f32::EPSILON {
        return left.original_seconds;
    }
    let t = (processed_seconds - left.processed_seconds) / den;
    left.original_seconds + (right.original_seconds - left.original_seconds) * t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_interpolates_between_anchors() {
        let map = TimelineMap::from_anchors(vec![
            TimelineAnchor {
                processed_seconds: 0.0,
                original_seconds: 0.0,
            },
            TimelineAnchor {
                processed_seconds: 10.0,
                original_seconds: 20.0,
            },
        ]);
        assert_eq!(map.map_processed_to_original_seconds(5.0), 10.0);
    }

    #[test]
    fn map_extrapolates_before_start_and_after_end() {
        let map = TimelineMap::from_anchors(vec![
            TimelineAnchor {
                processed_seconds: 2.0,
                original_seconds: 4.0,
            },
            TimelineAnchor {
                processed_seconds: 6.0,
                original_seconds: 8.0,
            },
        ]);
        assert_eq!(map.map_processed_to_original_seconds(0.0), 2.0);
        assert_eq!(map.map_processed_to_original_seconds(9.0), 11.0);
    }

    #[test]
    fn identity_map_preserves_values_beyond_one_second() {
        let map = TimelineMap::identity();
        assert_eq!(map.map_processed_to_original_seconds(2.5), 2.5);
        assert_eq!(map.map_processed_to_original_seconds(120.0), 120.0);
    }
}
