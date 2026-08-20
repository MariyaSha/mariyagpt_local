const form = document.getElementById(
    "setup-form"
);

const errorMessage = document.getElementById(
    "setup-error"
);

const modelSelect = document.getElementById(
    "model"
);

const hardwareInfo = document.getElementById(
    "hardware-info"
);

const downloadButton = document.getElementById(
    "download-model"
);

const startButton = document.getElementById(
    "start-chatting"
);

const progressContainer = document.getElementById(
    "download-progress"
);

const progressBar = document.getElementById(
    "download-progress-bar"
);

const downloadStatus = document.getElementById(
    "download-status"
);

const downloadAmount = document.getElementById(
    "download-amount"
);

const modelReady = document.getElementById(
    "model-ready"
);


let models = [];
let pollTimer = null;


function formatSize(sizeGb) {
    return `${Number(sizeGb).toFixed(2)} GB`;
}


function getSelectedModel() {
    return models.find(
        (model) =>
            model.id === modelSelect.value
    );
}


function stopPolling() {
    if (pollTimer !== null) {
        clearInterval(pollTimer);
        pollTimer = null;
    }
}


function resetDownloadInterface() {
    stopPolling();

    errorMessage.textContent = "";

    progressContainer.hidden = true;
    modelReady.hidden = true;

    progressBar.style.width = "0%";

    downloadStatus.textContent =
        "Preparing download...";

    downloadAmount.textContent =
        "0 GB / 0 GB";

    startButton.disabled = true;
}


function showModelState() {
    resetDownloadInterface();

    const model = getSelectedModel();

    if (!model) {
        downloadButton.disabled = true;
        downloadButton.textContent =
            "Select a model";

        return;
    }

    if (model.downloaded) {
        downloadButton.disabled = true;
        downloadButton.textContent =
            "Model Downloaded";

        modelReady.hidden = false;
        startButton.disabled = false;

        return;
    }

    downloadButton.disabled = false;

    downloadButton.textContent =
        `Download Model (~${formatSize(
            model.size_gb
        )})`;
}


async function loadHardware() {
    try {
        errorMessage.textContent = "";

        const response = await fetch(
            `${window.BACKEND_URL}/api/hardware`
        );

        if (!response.ok) {
            throw new Error(
                "Could not detect hardware."
            );
        }

        const data = await response.json();

        models = data.models;

        modelSelect.innerHTML = "";

        for (const model of models) {
            const option =
                document.createElement(
                    "option"
                );

            option.value = model.id;

            option.textContent =
                `${model.name} (~${formatSize(
                    model.size_gb
                )})`;

            modelSelect.appendChild(
                option
            );
        }

        if (models.length === 0) {
            throw new Error(
                "No compatible models are available."
            );
        }

        modelSelect.disabled = false;

        if (data.cuda) {
            hardwareInfo.textContent =
                `${data.device} ` +
                `(${data.vram_gb} GB VRAM)`;
        } else {
            hardwareInfo.textContent =
                "CPU (CUDA is not available)";
        }

        showModelState();

    } catch (error) {
        console.error(error);

        hardwareInfo.textContent =
            "Hardware detection failed.";

        modelSelect.disabled = true;
        downloadButton.disabled = true;
        startButton.disabled = true;

        errorMessage.textContent =
            error.message ||
            "Could not detect available hardware.";
    }
}


async function pollDownloadStatus() {
    try {
        const response = await fetch(
            `${window.BACKEND_URL}` +
            `/api/models/download-status`
        );

        if (!response.ok) {
            throw new Error(
                "Could not check download progress."
            );
        }

        const data = await response.json();

        const selectedModel =
            getSelectedModel();

        if (
            !selectedModel ||
            data.model !== selectedModel.id
        ) {
            return;
        }

        const percent = Math.max(
            0,
            Math.min(
                100,
                Number(data.percent) || 0
            )
        );

        progressBar.style.width =
            `${percent}%`;

        downloadAmount.textContent =
            `${formatSize(
                data.downloaded_gb
            )} / ${formatSize(
                data.total_gb
            )}`;

        if (
            data.status === "starting" ||
            data.status === "downloading"
        ) {
            progressContainer.hidden = false;

            downloadStatus.textContent =
                `Downloading... ${percent.toFixed(
                    1
                )}%`;

            downloadButton.disabled = true;
            downloadButton.textContent =
                "Downloading Model...";

            startButton.disabled = true;

            return;
        }

        if (data.status === "complete") {
            stopPolling();

            progressBar.style.width =
                "100%";

            downloadStatus.textContent =
                "Download complete";

            downloadButton.disabled = true;
            downloadButton.textContent =
                "Model Downloaded";

            selectedModel.downloaded = true;

            modelReady.hidden = false;
            startButton.disabled = false;

            return;
        }

        if (data.status === "error") {
            stopPolling();

            downloadButton.disabled = false;

            downloadButton.textContent =
                `Download Model (~${formatSize(
                    selectedModel.size_gb
                )})`;

            startButton.disabled = true;

            errorMessage.textContent =
                data.error ||
                "The model download failed.";
        }

    } catch (error) {
        console.error(error);

        stopPolling();

        downloadButton.disabled = false;
        startButton.disabled = true;

        errorMessage.textContent =
            error.message ||
            "Could not monitor the model download.";
    }
}


function startPolling() {
    stopPolling();

    pollDownloadStatus();

    pollTimer = setInterval(
        pollDownloadStatus,
        500
    );
}


modelSelect.addEventListener(
    "change",
    () => {
        showModelState();
    }
);


downloadButton.addEventListener(
    "click",
    async () => {
        const model = getSelectedModel();

        if (!model) {
            return;
        }

        resetDownloadInterface();

        downloadButton.disabled = true;
        downloadButton.textContent =
            "Starting Download...";

        progressContainer.hidden = false;

        downloadStatus.textContent =
            "Preparing download...";

        downloadAmount.textContent =
            `0.00 GB / ${formatSize(
                model.size_gb
            )}`;

        try {
            const response = await fetch(
                `${window.BACKEND_URL}` +
                `/api/models/download`,
                {
                    method: "POST",
                    headers: {
                        "Content-Type":
                            "application/json",
                    },
                    body: JSON.stringify({
                        model: model.id,
                    }),
                }
            );

            const data =
                await response.json();

            if (!response.ok) {
                throw new Error(
                    data.detail ||
                    "Could not start the download."
                );
            }

            if (data.status === "complete") {
                model.downloaded = true;

                showModelState();

                return;
            }

            startPolling();

        } catch (error) {
            console.error(error);

            progressContainer.hidden = true;

            downloadButton.disabled = false;

            downloadButton.textContent =
                `Download Model (~${formatSize(
                    model.size_gb
                )})`;

            errorMessage.textContent =
                error.message ||
                "Could not start the model download.";
        }
    }
);


form.addEventListener(
    "submit",
    async (event) => {
        event.preventDefault();

        errorMessage.textContent = "";

        const name = document
            .getElementById("name")
            .value
            .trim();

        const model =
            modelSelect.value;

        if (!name) {
            errorMessage.textContent =
                "Please enter your name.";

            return;
        }

        const selectedModel =
            getSelectedModel();

        if (
            !selectedModel ||
            !selectedModel.downloaded
        ) {
            errorMessage.textContent =
                "Download the selected model before starting MariyaGPT.";

            return;
        }

        startButton.disabled = true;
        startButton.textContent =
            "Starting...";

        try {
            const response = await fetch(
                `${window.BACKEND_URL}/api/setup`,
                {
                    method: "POST",
                    headers: {
                        "Content-Type":
                            "application/json",
                    },
                    body: JSON.stringify({
                        name: name,
                        model: model,
                    }),
                }
            );

            const data =
                await response.json();

            if (!response.ok) {
                throw new Error(
                    data.detail ||
                    "Setup failed."
                );
            }

            window.location.href =
                "/chat";

        } catch (error) {
            console.error(error);

            startButton.disabled = false;
            startButton.textContent =
                "Start Chatting";

            errorMessage.textContent =
                error.message ||
                "Could not connect to the backend.";
        }
    }
);


loadHardware();