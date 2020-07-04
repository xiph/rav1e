// Copyright (c) 2020, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

import React from 'react';

interface EncoderConfigDetailsProps {
    configStr: string
}

const EncoderConfigDetails: React.FC<EncoderConfigDetailsProps> = ({ configStr }) => {
    const styledStr = JSON.stringify(JSON.parse(configStr), null, 4);
    return (
        <details>
            <summary>Encoder Config</summary>
            <p style={{ whiteSpace: "pre-wrap" }}>{styledStr}</p>
        </details>
    )
};

export default EncoderConfigDetails;