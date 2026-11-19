"""GT rectification: remove hand-object penetration > 2 mm from MoCap/multi-view GT while staying consistent with
the cameras and smooth in time. Frames whose GT already penetrates <= 2 mm are never modified."""
