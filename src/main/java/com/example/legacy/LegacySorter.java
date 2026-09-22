package com.example.legacy;

import java.util.ArrayList;
import java.util.List;

/** 遗留代码：验收夹具（大文件，走骨架分支）。 */
public class LegacySorter {

    private boolean dirty;

    /** 过滤掉空值。 */
    public List<String> compact(List<String> input) {
        List<String> out = new ArrayList<>();
        for (String item : input) {
            if (item != null) {
                out.add(item);
            }
        }
        return out;
    }

    /** 代码转名称。 */
    public String nameOf(int code) {
        switch (code) {
            case 1:
                return "LOW";
            case 2:
                return "HIGH";
        }
        return "UNKNOWN";
    }

    /** 是否为空串。 */
    public boolean isBlank(String value) {
        if (value == null) {
            return true;
        } else {
            return false;
        }
    }

    /** 拆分。 */
    public String[] split(String raw) {
        String parts[] = raw.split(";");
        return parts;
    }

    /** 是否命中。 */
    public boolean hit(String code) {
        switch (code) {
            case "A":
                return true;
            case "B":
                this.dirty = true;
            default:
                return false;
        }
    }

    /** 归一化。 */
    public String normalize(String raw) {
        String result = raw;
        if (result == "")
            result = null;
        return result;
    }

    /** 是否脏。 */
    public boolean isDirty() {
        return dirty;
    }
}
